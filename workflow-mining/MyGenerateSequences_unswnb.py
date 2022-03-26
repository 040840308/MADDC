
import argparse, csv, datetime, gc, gzip, json, math, matplotlib, numpy, os, sys, time
matplotlib.use ('Agg')
import matplotlib.pyplot
import unswnb.util as util
import unswnb15
import unswnb15.key
import pandas as pd

attackers = ['175.45.176.0', '175.45.176.1', '175.45.176.2', '175.45.176.3']
victims = ['149.171.126.10', '149.171.126.11', '149.171.126.12', '149.171.126.13', '149.171.126.14',
           '149.171.126.15', '149.171.126.16', '149.171.126.17', '149.171.126.18', '149.171.126.19']
normals = ['149.171.126.0', '149.171.126.1', '149.171.126.2', '149.171.126.3', '149.171.126.4',
           '149.171.126.5', '149.171.126.6', '149.171.126.7', '149.171.126.8', '149.171.126.9'] # majorly inbound
senders = ['59.166.0.0', '59.166.0.1', '59.166.0.2', '59.166.0.3', '59.166.0.4',
           '59.166.0.5', '59.166.0.6', '59.166.0.7', '59.166.0.8', '59.166.0.9'] # majorly outbound

###############################################################
##                           Main                            ##
###############################################################
def main ():
    timestamp = datetime.datetime.now ()
    operation = {
        'key': [LogKeyStatistics, ],
        'generatesequence': [GenerateSequence, ],
        }
    parser = argparse.ArgumentParser (description='Experiment based on UNSW-NB15 dataset')
    parser.add_argument ('operation', help='Specify an operation (' + ', '.join (operation.keys ()) + ')')
    # Anomaly Detection
    parser.add_argument ('-i', '--input', help='Specify a input file or directory that contains unswnbData (e.g., mydata/)', default='unswnbData')
    parser.add_argument ('-o', '--output', help='Specify an output file to write to', default=timestamp.strftime ('%Y_%m_%d_%H_%M_%S'))
    parser.add_argument ('--train-file', help='Specify a filename for training', default='flow_from_this.csv')
    parser.add_argument ('--test-file', help='Specify a filename for testing', default='flow_to_this.csv')
    parser.add_argument ('--seqlen', help='Hyperparameter', type=int, default=10)
    parser.add_argument ('--logkeys', help='Logkeys Scenarios (see scenario index in key.py)', type=int, default=0)
    parser.add_argument ('--stats-step', help='Step for stats (default=0.01)', type=float, default=0.01)
    parser.add_argument ('--window-size', help='Specify the number of minutes for each sequence', type=int, default=15)
    parser.add_argument ('--key-divisor', help='Speicfy the number of key divisor', type=int, default=100)
    parser.add_argument ('--label-size', help='Specify the number of attack category for anomlies', type=int, default=1)
    parser.add_argument ('--check-sequences', help='Specify a file of subjects to check')
    # Parser
    args = parser.parse_args ()
    for op in operation [args.operation]:
        gc.collect ()
        op (args)


def LogKeyStatistics(args):
    def getKeyset(iplist):
        ret = {}
        for ip in iplist:
            with open(os.path.join(args.input, ip, 'flows.csv'), 'rt', encoding='utf-8') as fin:
                csvfin = csv.reader(fin, delimiter=',')
                for line in csvfin:
                    # linestr = unswnb15.key.getDirectionFromLine (line, ip) + ',' + unswnb15.key.getKeyFromLine (line, args.logkeys, args.key_divisor)
                    linestr = unswnb15.key.getKeyFromLine(line, args.logkeys, args.key_divisor)
                    ret.setdefault(linestr, 0)
                    ret[linestr] += 1
        return ret

    k_attackers, k_victims, k_normals, k_senders = getKeyset(attackers), getKeyset(victims), getKeyset(
        normals), getKeyset(senders)
    s_attackers, s_victims, s_normals, s_senders = set(k_attackers.keys()), set(k_victims.keys()), set(
        k_normals.keys()), set(k_senders.keys())

    print('# of Logkeys from All:', len(s_attackers | s_victims | s_normals | s_senders))
    print('# of Logkeys from Attackers:', len(s_attackers))
    print('# of Logkeys from Victims:', len(s_victims))
    print('# of Logkeys from Senders:', len(s_senders))
    print('# of Logkeys from Normal Receivers:', len(s_normals))
    print('# of LogKeys that are Abnormal:', len(s_attackers | s_victims))
    print('Intersection (abnormal & normal):', len((s_attackers | s_victims) & (s_senders | s_normals)))
    logkeys = {}
    #for party in k_attackers, k_victims, k_normals, k_senders:
    for party in [k_senders]:
        for key in party:
            logkeys.setdefault(key, 0)
            logkeys[key] += party[key]
    summention = sum(logkeys.values())
    logkeys = sorted(logkeys.items(), reverse=True, key=lambda item: item[1])
    accumulate = 0
    keysmap={}
    for index, (key, occurrence) in enumerate(logkeys):
        keysmap[str(key)]=index
        accumulate += occurrence
        print('{:04d}'.format(index + 1), '{:05.02f}'.format(accumulate * 100 / summention), occurrence,
              '{:05.02f}'.format(occurrence * 100 / summention), key)

    with open("TotalKeysMap_unswnb.json", 'w', encoding='utf-8') as keyf:
        json.dump(keysmap,keyf)


def GenerateSequence (args):
    # define how to read sequences from file
    def readSequences(ip, filename):
        sequence = {}
        label = {}
        with open(os.path.join(args.input, ip, filename), 'rt',encoding='utf-8') as fin:
            csvfin = csv.reader(fin, delimiter=',')
            for line in csvfin:
                #print(line)
                datetime = unswnb15.key.getDateTimeFromLine(line)
                srcip = line[unswnb15.key.srcip]
                dstip = line[unswnb15.key.dstip]
                dstport = line[unswnb15.key.dsport]
                svcport = dstport
                # try: svcport = line [unswnb15.key.proto] if int (dstport) > 1024 else dstport
                # except: pass
                # subject = '-'.join (['from', srcip, 'to', dstip, ':', dstport])
                # subject = '-'.join (['from', srcip, 'to', dstip, ':', svcport])
                #print(args.window_size)
                subject = '-'.join(['from', srcip, 'to', dstip, 'on', str(datetime.day), str(datetime.hour),
                                    str(datetime.minute // args.window_size)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 60)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 30)])
                # subject = '-'.join (['from', srcip, 'to', dstip, 'on', str (datetime.day), str (datetime.hour), str (datetime.minute // 15)])
                slabel = unswnb15.key.getLabelFromLine(line)
                skeystr = unswnb15.key.getKeyFromLine(line, args.logkeys, args.key_divisor)
                if subject not in sequence: sequence[subject] = list()
                sequence[subject].append(skeystr)
                if subject not in label: label[subject] = list()
                label[subject].append(slabel)
        ret = []
        for subject in sequence:
            notNoneLabels = [l for l in label[subject] if l is not None]
            ret.append((
                sequence[subject],
                subject,
                ','.join(sorted(set(notNoneLabels))),
                len(notNoneLabels)))
        return ret

    # build training dataset
    trainips = senders
    bar = util.ProgressBar('Read Normal Sequences for Training', len(trainips))
    #load sequence keys
    with open("TotalKeysMap_unswnb.json", 'r') as keyf:
        keysmap=json.load(keyf)

    # make statistics the one-hot encodes
    event_onehot = set()
    #statistics of sequence lines
    sequence_number=0
    train_sequence_number = 0
    test_normal_sequence_number = 0
    test_abnormal_sequence_number = 0
    min_len=3 # not to consider too short sequences
    #pad:0, start:1, end:2, unknown:3
    for idx, ip in enumerate(trainips):
        dstdirname = os.path.join(args.output, "Train")
        os.makedirs(dstdirname, exist_ok=True)
        dstdirname=os.path.join(dstdirname,ip)
        os.makedirs(dstdirname, exist_ok=True)
        bar.update(idx + 1)
        sequencesDict={}
        try:
            for sequence, description, label, attacks in readSequences(ip, args.train_file):
                #onehot = [0] * len(keysmap)
                newsequence=[]
                for s in sequence:
                    newsequence.append(str(keysmap[s] + 4))
                #newsequence.append(str(2))
                if len(newsequence)<=min_len:
                    continue # skip these sequences, it is too shot
                #for s1 in newsequence:
                #    onehot[int(s1)] = 1
                #event_onehot.add(tuple(onehot))
                sequence_number +=1
                train_sequence_number += 1
                #print(newsequence)
                description=description+"*"+label
                sequencesDict[description]=' '.join(newsequence)
        except KeyboardInterrupt as e:
            print(e); exit(0)
        except Exception as e:
            print(e); continue
        #saved to csv
        df = pd.DataFrame(list(sequencesDict.items()),columns=['description', 'sequences'])
        saved_CSV=os.path.join(dstdirname,"sequences.csv")
        df.to_csv(saved_CSV, index=None)
        #saved to a sequence file
        sequenceData = df['sequences'].to_list()
        saved_TXT = os.path.join(dstdirname, "sequences.txt")
        f = open(saved_TXT, "w")  # 利用追加模式,参数从w替换为a即可
        for seq in sequenceData:
            f.write("{}\n".format(seq))
        f.close()
        sequence_json_file=os.path.join(dstdirname, "sequences.json")
        with open(sequence_json_file, "w") as f:
            json.dump(sequencesDict, f)
            f.close()


    bar.finish()

    # build testing dataset from normals
    trainips = normals
    bar = util.ProgressBar('Read Test Normal Sequences for Training', len(trainips))
    # load sequence keys
    with open("TotalKeysMap_unswnb.json", 'r') as keyf:
        keysmap = json.load(keyf)

    for idx, ip in enumerate(trainips):
        dstdirname = os.path.join(args.output, "Test_Normal")
        #if ip=='149.171.126.6':
        #    print(ip)
        os.makedirs(dstdirname, exist_ok=True)
        dstdirname = os.path.join(dstdirname, ip)
        os.makedirs(dstdirname, exist_ok=True)
        bar.update(idx + 1)
        sequencesDict = {}
        try:
            for sequence, description, label, attacks in readSequences(ip, args.test_file):
                #onehot = [0] * len(keysmap)
                #newsequence = [str(keysmap[s]) for s in sequence]
                newsequence = []
                for s in sequence:
                    if s in keysmap:
                        newsequence.append(str(keysmap[s] + 4))
                    else:
                        newsequence.append(str(3))
                #newsequence.append(str(2))
                if len(newsequence)<=min_len:
                    continue # skip these sequences, it is too shot
                #for s1 in newsequence:
                #    onehot[int(s1)] = 1
                #event_onehot.add(tuple(onehot))
                sequence_number += 1
                test_normal_sequence_number += 1
                # print(newsequence)
                description = description + "*" + label
                sequencesDict[description] = ' '.join(newsequence)
        except KeyboardInterrupt as e:
            print(e); exit(0)
        except Exception as e:
            print(e); continue
        # saved to csv
        df = pd.DataFrame(list(sequencesDict.items()), columns=['description', 'sequences'])
        saved_CSV = os.path.join(dstdirname, "sequences.csv")
        df.to_csv(saved_CSV, index=None)
        # saved to a sequence file
        sequenceData = df['sequences'].to_list()
        saved_TXT = os.path.join(dstdirname, "sequences.txt")
        f = open(saved_TXT, "w")  # 利用追加模式,参数从w替换为a即可
        for seq in sequenceData:
            f.write("{}\n".format(seq))
        f.close()
        sequence_json_file = os.path.join(dstdirname, "sequences.json")
        with open(sequence_json_file, "w") as f:
            json.dump(sequencesDict, f)
            f.close()

    bar.finish()

    # build testing dataset from victims
    trainips = victims
    bar = util.ProgressBar('Read Test Abnormal Sequences for Training', len(trainips))
    # load sequence keys
    with open("TotalKeysMap_unswnb.json", 'r') as keyf:
        keysmap = json.load(keyf)

    for idx, ip in enumerate(trainips):
        dstdirname = os.path.join(args.output, "Test_Abnormal")
        os.makedirs(dstdirname, exist_ok=True)
        dstdirname = os.path.join(dstdirname, ip)
        os.makedirs(dstdirname, exist_ok=True)
        bar.update(idx + 1)
        sequencesDict = {}
        try:
            for sequence, description, label, attacks in readSequences(ip, args.test_file):
                #onehot = [0] * len(keysmap)
                newsequence = []
                for s in sequence:
                    if s in keysmap:
                        newsequence.append(str(keysmap[s] + 4))
                    else:
                        newsequence.append(str(3))
                #newsequence.append(str(2))
                if len(newsequence)<=min_len:
                    continue # skip these sequences, it is too shot
                #for s1 in newsequence:
                #    onehot[int(s1)] = 1
                #event_onehot.add(tuple(onehot))
                sequence_number += 1
                test_abnormal_sequence_number += 1
                # print(newsequence)
                description = description + "*" + label
                sequencesDict[description] = ' '.join(newsequence)
        except KeyboardInterrupt as e:
            print(e); exit(0)
        except Exception as e:
            print(e); continue
        # saved to csv
        df = pd.DataFrame(list(sequencesDict.items()), columns=['description', 'sequences'])
        saved_CSV = os.path.join(dstdirname, "sequences.csv")
        df.to_csv(saved_CSV, index=None)
        # saved to a sequence file
        sequenceData = df['sequences'].to_list()
        saved_TXT = os.path.join(dstdirname, "sequences.txt")
        f = open(saved_TXT, "w")  # 利用追加模式,参数从w替换为a即可
        for seq in sequenceData:
            f.write("{}\n".format(seq))
        f.close()
        sequence_json_file = os.path.join(dstdirname, "sequences.json")
        with open(sequence_json_file, "w") as f:
            json.dump(sequencesDict, f)
            f.close()

    bar.finish()

    #print("Total unique one-hot encode: %d" % len(event_onehot))
    print("Total Sequence number: %d" % sequence_number)
    print("Total Train Sequence number: %d" % train_sequence_number)
    print("Total Test Normal Sequence number: %d" % test_normal_sequence_number)
    print("Total Test Abnormal Sequence number: %d" % test_abnormal_sequence_number)



if __name__ == '__main__': main ()