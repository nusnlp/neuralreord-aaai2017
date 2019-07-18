#!/usr/bin/python
import sys, re
import os.path
import argparse
from dependency_input import Dependency
from operator import itemgetter

PUNCTS=set(['PU', 'PUNCT', '``', ',', ':', '.', "''", '-LRB-', '-RRB-'])

def smart_open(fname, mode = 'r'):
    if fname.endswith('.gz'):
        import gzip
        # Using max compression (9) by default seems to be slow.
        # Let's try using the fastest.
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode)

def has_punct(tree, sublist):
    # sublist is list of child indices
    for e in sublist:
        if tree.getPostag(e) in PUNCTS:
            return True
    return False

def extract_inbetween(children, start, end, headLeft=None, headRight=None):
    # headLeft is not None when there is head and it is on the left
    # headRight is not None when there is head and it is on the right
    assert headLeft == None or headRight == None, "Head must be EITHER left or right, NOT BOTH!"
    if headLeft != None:
        return [c for c in children[start:end] if c > headLeft]
    elif headRight != None:
        return [c for c in children[start:end] if c < headRight]
    else:
        return children[start:end]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #argparser.add_argument("--with_others", dest="with_others", action='store_true', help="Include other features other than word pair links.")
    argparser.add_argument("argv", metavar='A', nargs='+')
    args = argparser.parse_args()

    vcbName = args.argv[0]
    depName = args.argv[1]
    aliName = args.argv[2]
    outParName = args.argv[3]
    outSibName = args.argv[4]

    depFile = smart_open(depName, 'r')
    aliFile = smart_open(aliName, 'r')
    outParFile = smart_open(outParName, 'w')
    outSibFile = smart_open(outSibName, 'w')
    
    dotCutoff = 500
    dpl = 100
    nlCutoff = dotCutoff * dpl

    vcbFile = smart_open(vcbName, 'r')
    vcb = set()
    print >> sys.stderr, 'Loading word vocabulary ...'
    for w in vcbFile:
        w = w.decode('utf-8').strip()
        vcb.add(w.encode('utf-8'))
    vcbFile.close()
    print >> sys.stderr, "Vocab size = %d words." % len(vcb)

    lCount = 0
    while True:
        depTree = Dependency(depFile)
        treeLen = depTree.getLength()
        if treeLen == 0:
            break

        alignLine = aliFile.readline().strip()
        if alignLine == '':
            continue
        alignToks = [tuple([int(t) for t in s.split('-')]) for s in alignLine.split(' ')]
        
        # 1. Normalizing alignment, following (Bizazza & Federico, 2013)
        f2e = {}
        tgtMax = -1
        for a in alignToks:
            if a[1] > tgtMax:
                tgtMax = a[1]
            if a[0] in f2e:
                f2e[a[0]].append(a[1])
            else:
                f2e[a[0]] = [a[1]]
        tgtLen = tgtMax + 1
        
        # averaging multi-alignment
        f2e_avg = {}
        for f in sorted(f2e.keys()):
            f2e_avg[f] = sum(f2e[f]) / float(len(f2e[f]))
        
        # filling in the blank
        for i in xrange(treeLen):
            if i not in f2e_avg: # unaligned source
                # find left anchor
                j = i-1
                found = False
                while j >= 0 and not found:
                    if j in f2e_avg:
                        found = True
                    else:
                        j -= 1
                leftAnchor = f2e_avg[j] if j >= 0 else -1.0
                
                # find right anchor
                j = i+1
                found = False
                while j < treeLen and not found:
                    if j in f2e_avg:
                        found = True
                    else:
                        j += 1
                rightAnchor = f2e_avg[j] if j < treeLen else tgtLen
                f2e_avg[i] = (leftAnchor + rightAnchor) / 2.0
        
        for node in xrange(treeLen):
            children = sorted(depTree.getChildren(node).keys())
            childrenFeats = []
            hWord = depTree.getWord(node).encode('utf-8')
            hItem = hWord if hWord in vcb else "<UNK>"
            hPos = depTree.getPostag(node).encode('utf-8') + '<T>'
            ghLabel = depTree.getLabel(node).encode('utf-8') + '<L>'
            for i in xrange(len(children)):
                # 2. generate example for head-child relations
                left, right = (node, children[i]) if node < children[i] else (children[i], node)
                cWord = depTree.getWord(children[i]).encode('utf-8')
                cItem = cWord if cWord in vcb else "<UNK>"
                cPos = depTree.getPostag(children[i]).encode('utf-8') + '<T>'
                hcLabel = depTree.getLabel(children[i]).encode('utf-8') + '<L>'

                lItem = cItem if children[i] < node else "<NULL>"
                lPos = cPos if children[i] < node else "<NULL>"
                lLabel = hcLabel if children[i] < node else "<NULL>"

                rItem = cItem if children[i] > node else "<NULL>"
                rPos = cPos if children[i] > node else "<NULL>"
                rLabel = hcLabel if children[i] > node else "<NULL>"
                
                inBetween = extract_inbetween(children, None, i, headLeft=node) if node < children[i] else extract_inbetween(children, i+1, None, headRight=node)
                hasPunctStr = "1<,>" if has_punct(depTree, inBetween) else "0<,>"

                binDist = -2 if children[i] < node else 2
                if binDist == -2 and (i+1 >= len(children) or children[i+1] > node):
                    binDist += 1
                elif binDist == 2 and (i-1 < 0 or children[i-1] < node):
                    binDist -= 1
                oriNum = 1 if f2e_avg[right] < f2e_avg[left] else 0 # 1: swapped; 0: in-order
                
                featStr = "{0} {1} {2} {3:d}<D>".format(cItem, cPos, hcLabel, binDist)
                childrenFeats.append(featStr)

                print >> outParFile, "{:d} {} {} {} {} {} {} {} {} {} {:d}<D> {}".format(oriNum, hItem, hPos, ghLabel, lItem, lPos, lLabel, rItem, rPos, rLabel, binDist, hasPunctStr)

            for i in xrange(len(children)):
                # 3. generate example for sibling relations
                for j in range(i + 1, len(children)):
                    leftFeats, rightFeats = (childrenFeats[i], childrenFeats[j])
                    (left, right) = (children[i], children[j])
                    leftLabel = depTree.getLabel(left).encode('utf-8')
                    rightLabel = depTree.getLabel(right).encode('utf-8')
                    oriNum = 1 if f2e_avg[right] < f2e_avg[left] else 0 # 1: swapped; 0: in-order

                    inBetween = extract_inbetween(children, i+1, j)
                    hasPunctStr = "1<,>" if has_punct(depTree, inBetween) else "0<,>"

                    print >> outSibFile, "{0:d} {1} {2} {3} {4} {5}".format(oriNum, hItem, hPos, childrenFeats[i], childrenFeats[j], hasPunctStr)

        # trace progress
        lCount += 1
        if lCount % dotCutoff == 0:
            sys.stderr.write('.')
        if lCount % nlCutoff == 0:
            print >> sys.stderr, "%d" % lCount

    print >> sys.stderr, "[%d sentences in total]" % lCount
