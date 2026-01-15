import argparse
import time
import torch.optim as optim
from torch.autograd import Variable
import utils
from modules.mcc import MinimumClassConfusionLoss
from modules.teacher import EMATeacher
from UDNNET import UDNNET
from utils import *
from basenet import *
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--pseudo_label_weight', default="prob")
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--halfwidth', default=3, type=int)
parser.add_argument('--temperature', default=6.0,
                        type=float, help='parameter temperature scaling')

args = parser.parse_args()
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
num_epoch = args.num_epoch
batch_size = args.batch_size
lr = args.lr
num_k = args.num_k
use_gpu = torch.cuda.is_available()
# nDataSet = 10
num_classes = 7
N_BANDS = 270
HalfWidth = args.halfwidth
seeds = [1838,1786,1427,1589,1353,1896,1599,1868,1700,1814]
nDataSet = len(seeds)
FM = 64
b = 0.1  # nsd loss weight

g = {}
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, num_classes])
k = np.zeros([nDataSet, 1])
# best_predict_all = []
best_acc_all = 0.0
best_predict_all = 0
best_test_acc = 0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
for iDataSet in range(nDataSet):
    data_path_s = 'data/YRD/yrd_nc12.mat'
    label_path_s = 'data/YRD/yrd_nc12_7gt.mat'
    data_path_t = 'data/YRD/yrd_nc13.mat'
    label_path_t = 'data/YRD/yrd_nc13_7gt.mat'

    data_s, label_s = utils.load_data_YRD(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_YRD(data_path_t, label_path_t)
    pca_n = 2
    radius = 0.00009
    data_s, data_t = ILDA(data_s, data_t, pca_n, radius)
    print('#######################idataset######################## ', iDataSet)
    utils.seed_everything(seeds[iDataSet], use_deterministic=True)

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)

    testID, testX, testY, Gr, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_tar_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    len_src_loader = len(train_loader)
    len_tar_train_loader = len(train_tar_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_train_dataset = len(train_tar_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    G = UDNNET(input_channels=N_BANDS, num_classes=num_classes, dim=FM * 4).to(DEV)

    F1 = ResClassifier(num_classes, FM * 4, training=True).to(DEV)
    F2 = ResClassifier(num_classes, FM * 4, training=True).to(DEV)
    F1.apply(weights_init)
    F2.apply(weights_init)

    F1_teacher = EMATeacher(G, F1, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(DEV)

    optimizer_g = optim.Adam(G.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.9)
    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=0.01,
                            weight_decay=0.005)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)


    def train(ep, train_loader, train_tar_loader):
        iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
        criterion = nn.CrossEntropyLoss().cuda()
        G.train()
        F1.train()
        F2.train()
        num_iter = len_src_loader
        m = 2
        for batch_idx in range(1, num_iter):
            torch.use_deterministic_algorithms(False)
            if batch_idx % len(train_tar_loader) == 0:
                iter_target = iter(train_tar_loader)
            data_source, label_source = iter_source.__next__()
            data_target, _ = iter_target.__next__()

            source_data0 = utils.radiation_noise(data_source, alpha_range=(0.5, 1.5), beta=0.001).type(torch.FloatTensor)
            target_data0 = utils.radiation_noise(data_target, alpha_range=(0.5, 1.5), beta=0.001).type(torch.FloatTensor)

            data1, target1 = data_source.cuda(), label_source.cuda()
            data2 = data_target.cuda()
            source_data0 = source_data0.cuda()
            target_data0 = target_data0.cuda()
            # when pretraining network source only
            data = Variable(torch.cat((data1,source_data0, data2), 0))
            target1 = Variable(torch.cat((target1, target1)))

            F1_teacher.update_weights(G, F1, ep * num_iter + batch_idx)
            F1_pseudo_label_t, F1_pseudo_prob_t = F1_teacher(data2)

            # Step A train all networks to minimize loss on source

            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(data)

            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[0][:m * batch_size, :]
            output_s2 = output2[0][:m * batch_size, :]
            output_t1 = output1[0][batch_size * m:, :]
            output_t2 = output2[0][batch_size * m:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)

            loss1 = criterion(output_s1, target1.long())
            loss2 = criterion(output_s2, target1.long())

            loss_nsd = 0
            output = G(target_data0)
            y_t_A_f1, _ = F1(output)
            if F1_teacher.pseudo_label_weight is not None:
                ce = F.cross_entropy(y_t_A_f1, F1_pseudo_label_t, reduction='none').float()
                loss_nsd += torch.mean(F1_pseudo_prob_t * ce)
            else:
                loss_nsd += F.cross_entropy(y_t_A_f1, F1_pseudo_label_t)

            mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
            all_loss = loss1 + loss2 + loss_nsd * b + mcc_loss_value

            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B train classifier to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(data)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[0][:m * batch_size, :]
            output_s2 = output2[0][:m * batch_size, :]
            output_t1 = output1[0][batch_size * m:, :]
            output_t2 = output2[0][batch_size * m:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)
            loss1 = criterion(output_s1, target1.long())
            loss2 = criterion(output_s2, target1.long())

            loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
            mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))

            F_loss = loss1 + loss2 - loss_dis + mcc_loss_value

            F_loss.backward()
            optimizer_f.step()
            # Step C train genrator to minimize discrepancy
            for i in range(num_k):
                optimizer_g.zero_grad()
                output = G(data)
                output1 = F1(output)
                output2 = F2(output)

                output_s1 = output1[0][:m * batch_size, :]
                output_s2 = output2[0][:m * batch_size, :]
                output_t1 = output1[0][batch_size * m:, :]
                output_t2 = output2[0][batch_size * m:, :]

                loss1 = criterion(output_s1, target1.long())
                loss2 = criterion(output_s2, target1.long())
                output_t1 = F.softmax(output_t1, dim=1)
                output_t2 = F.softmax(output_t2, dim=1)

                mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
                loss_dis = torch.mean(torch.abs(output_t1 - output_t2)) + mcc_loss_value

                loss_dis.backward()
                optimizer_g.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f}'.format(
                        ep, batch_idx * len(data), args.batch_size * len(train_loader),
                            100. * batch_idx / len(train_loader), loss1.item(), loss2.item(), loss_dis.item(),
                        ), seeds[iDataSet])

            if batch_idx == 1 and ep > 1:
                G.train()
                F1.train()
                F2.train()


    def test(test_loader):
        G.eval()
        F1.eval()
        F2.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0
        predict = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)
        predict1 = np.array([], dtype=np.int64)
        labels1 = np.array([], dtype=np.int64)
        pred1_list, pred2_list, label_list, outputdata = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target
                data1, target1 = Variable(data), Variable(target2)
                output = G(data1)
                outputdata.append(output.cpu().numpy())
                output1 = F1(output)
                output2 = F2(output)
                test_loss += F.nll_loss(output1, target1.long()).item()

                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.eq(target1.data).cpu().sum()
                pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(target1.data).cpu().sum()
                k = target1.data.size()[0]
                pred1_list.append(pred1.cpu().numpy())
                pred2_list.append(pred2.cpu().numpy())
                predict = np.append(predict, pred1.cpu().numpy())
                predict1 = np.append(predict1, pred2.cpu().numpy())
                labels = np.append(labels, target.cpu().numpy())
                labels1 = np.append(labels1, target.cpu().numpy())
                label_list.append(target2.cpu().numpy())
                size += k
                acc1 = 100. * float(correct) / float(size)
                acc2 = 100. * float(correct2) / float(size)

            test_loss = test_loss
            test_loss /= len(test_loader)  # loss function already averages over batch size
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%)\n'.format(
                test_loss, correct, len_tar_dataset,
                100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset))
            # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
            value = max(100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset)

            if acc1 > acc2:
                return value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata, label_list, predict, labels
            else:
                return value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata, label_list, predict1, labels1


    torch.cuda.synchronize()
    train_start = time.time()
    value = 0
    for ep in range(1, num_epoch + 1):
        train(ep, train_loader, train_tar_loader)
        if ep > 0:
            test_start = time.time()
            value1, pred1_list, pred2_list, label_list, acc1, acc2, outputdata_target, target_label, predict, labels = test(
                test_loader)
            test_end = time.time()
            if value < value1:
                value = value1
                acc[iDataSet] = value
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float32)
                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                if value >= best_test_acc:
                    # torch.save(G, "checkpoints/YRD/G-{}.pt".format(value))
                    # torch.save(F1, "checkpoints/YRD/F1-{}.pt".format(value))
                    # torch.save(F2, "checkpoints/YRD/F2-{}.pt".format(value))
                    best_test_acc = value
                    best_predict_all = predict
                    best_G, best_RandPerm, best_Row, best_Column = Gr, RandPerm, Row, Column

        scheduler.step()
    print('Best test accuracy: {}'.format(value))
    g[seeds[iDataSet]] = value
    torch.cuda.synchronize()
    train_end = time.time()

for i in range(nDataSet):
    print(seeds[i] , g[seeds[i]])
AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - test_start))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(num_classes):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))