# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
import random
import math
import time
import codecs
import os


class HSMMWordSegm():
    MAX_LEN = 15
    AVE_LEN = 8

    def __init__(self, nclass ):
        self.num_class = nclass
        self.segm_class = {}
        self.segm_sequences = []
        self.trans_prob = np.ones( (nclass,nclass) )
        self.trans_prob_bos = np.ones( nclass )
        self.trans_prob_eos = np.ones( nclass )
        
        
    # クラスcに分節を追加
    def add_segm(self, c, segm ):
        self.segm_class[id(segm)] = c
        for i in range(len(segm)-1):
            o1 = segm[i]
            o2 = segm[i+1]
            self.obs_trans_count[c][o1][o2] += 1
            self.obs_count[c][o1] += 1
        
    # クラスcから分節を削除
    def delete_segm(self, c, segm):
        self.segm_class.pop(id(segm))
        for i in range(len(segm)-1):
            o1 = segm[i]
            o2 = segm[i+1]
            self.obs_trans_count[c][o1][o2] -= 1
            self.obs_count[c][o1] -= 1


    def load_data(self, filename ):
        # 観測シーケンス
        self.sequences = [ [ int(i) for i in line.split() ] for line in open( filename ).readlines()]
        
        # 観測の種類
        print( self.sequences )
        self.num_obs = int(np.max( [ np.max(s) for s in self.sequences] )+1)
        
        # 観測の遷移確率を計算するためのパラメータ
        self.obs_trans_count = np.zeros((self.num_class,self.num_obs,self.num_obs) )
        self.obs_count = np.zeros( (self.num_class,self.num_obs) )

        # ランダムに分節化        
        self.segm_sequences = []
        for seq in self.sequences:
            segms = []

            i = 0
            while i<len(seq):
                # ランダムに切る
                length = random.randint(1,self.MAX_LEN)

                if i+length>=len(seq):
                    length = len(seq)-i

                segms.append( seq[i:i+length] )

                i+=length

            self.segm_sequences.append( segms )

            # ランダムに割り振る
            for i,segm in enumerate(segms):
                c = random.randint(0,self.num_class-1)
                self.add_segm( c, segm )

        # 遷移確率更新
        self.calc_trans_prob()

    def calc_output_prob(self, c , segm ):
        # 長さの制約
        L = len(segm)
        prior = (self.AVE_LEN**L) * math.exp( -self.AVE_LEN ) / math.factorial(L)
        
        # クラスcの遷移確率から生成される確率
        p = prior
        for i in range(len(segm)-1):
            o1 = segm[i]
            o2 = segm[i+1]
            p *= (self.obs_trans_count[c][o1][o2] + 0.1)/(self.obs_count[c][o1] + self.num_obs*0.1 )
        

        return p

    def forward_filtering(self, sequence ):
        T = len(sequence)
        a = np.zeros( (len(sequence), self.MAX_LEN, self.num_class) )                            # 前向き確率

        for t in range(T):
            for k in range(self.MAX_LEN):
                if t-k<0:
                    break

                for c in range(self.num_class):
                    out_prob = self.calc_output_prob( c , sequence[t-k:t+1] )

                    # 遷移確率
                    tt = t-k-1
                    if tt>=0:
                        for kk in range(self.MAX_LEN):
                            for cc in range(self.num_class):
                                a[t,k,c] += a[tt,kk,cc] * self.trans_prob[cc, c]
                        a[t,k,c] *= out_prob
                    else:
                        # 最初の単語
                        a[t,k,c] = out_prob * self.trans_prob_bos[c]

                    # 最後の単語の場合
                    if t==T-1:
                        a[t,k,c] *= self.trans_prob_eos[c]

        return a

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

    def backward_sampling(self, a, sequence, use_max_path=False):
        T = a.shape[0]
        t = T-1

        segms = []
        classes = []

        c = -1
        while True:

            # 状態cへ遷移する確率
            if c==-1:
                trans = np.ones( self.num_class )
            else:
                trans = self.trans_prob[:,c]

            if use_max_path:
                idx = np.argmax( (a[t]*trans).reshape( self.MAX_LEN*self.num_class ) )
            else:
                idx = self.sample_idx( (a[t]*trans).reshape( self.MAX_LEN*self.num_class ) )


            k = int(idx/self.num_class)
            c = idx % self.num_class

            segm = sequence[t-k:t+1]

            segms.insert( 0, segm )
            classes.insert( 0, c )

            t = t-k-1

            if t<0:
                break

        return segms, classes


    def calc_trans_prob( self ):
        self.trans_prob = np.zeros( (self.num_class,self.num_class) ) + 1.0
        self.trans_prob_bos = np.zeros( self.num_class ) + 1.0
        self.trans_prob_eos = np.zeros( self.num_class ) + 1.0
        

        # 数え上げる
        for n,segms in enumerate(self.segm_sequences):
            try:
                # BOS
                c = self.segm_class[ id(segms[0]) ]
                self.trans_prob_bos[c] += 1
            except KeyError as e:
                # gibss samplingで除かれているものは無視
                continue

            for i in range(1,len(segms)):
                cc = self.segm_class[ id(segms[i-1]) ]
                c = self.segm_class[ id(segms[i]) ]

                self.trans_prob[cc,c] += 1.0

            # EOS
            c = self.segm_class[ id(segms[-1]) ]
            self.trans_prob_eos[c] += 1

        # 正規化
        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.num_class,1)
        self.trans_prob_bos = self.trans_prob_bos / self.trans_prob_bos.sum()
        self.trans_prob_eos = self.trans_prob_eos / self.trans_prob_eos.sum()


    def print_result(self):
        print ("-------------------------------")
        for segms in self.segm_sequences:
            for s in segms:
                print( s, ":", self.segm_class[id(s)] )
            print ("------------")
        print("")

        for c in range(self.num_class):
            print( "class", c )
            print( self.obs_trans_count[c] )
            print()


    def learn(self,use_max_path=False):
        for i in range(len(self.sequences)):
            seq = self.sequences[i]
            segms = self.segm_sequences[i]

            # 学習データから削除
            for s in segms:
                c = self.segm_class[id(s)]
                self.delete_segm(c, s)

            # 遷移確率更新
            self.calc_trans_prob()

            # foward確率計算
            a = self.forward_filtering( seq )

            # backward sampling
            # segms: 分節化されたシーケンス
            # classes: 各分節が分類されたクラス
            segms, classes = self.backward_sampling( a, seq, use_max_path )

            # パラメータ更新
            self.segm_sequences[i] = segms
            for s,c in zip( segms, classes ):
                self.add_segm( c, s )

            # 遷移確率更新
            self.calc_trans_prob()

        return
    
    def calc_loglik(self):
        lik = 0
        for segms in self.segm_sequences:
            for i in range(len(segms)-1):
                s1 = segms[0]
                s2 = segms[1]
                
                c1 = self.segm_class[id(s1)]
                c2 = self.segm_class[id(s2)]
                
                lik += math.log( self.calc_output_prob(c1, s1) * self.trans_prob[c1,c2] )
            # BOS
            c1 = self.segm_class[id(segms[0])]
            lik += math.log( self.trans_prob_bos[c1] )
            
            # EOS
            s1 = segms[-1]
            c1 = self.segm_class[id(s1)]
            
            lik += math.log( self.calc_output_prob(c1, s1) * self.trans_prob_eos[c1] )

        return lik            

    def save_result(self, dir ):
        if not os.path.exists(dir):
            os.mkdir(dir)

        for c in range(self.num_class):
            fname = os.path.join( dir , "trans_count_%03d.txt" %c )
            np.savetxt( fname, self.obs_trans_count[c] )

        path = os.path.join( dir , "result.txt" )
        f = codecs.open( path ,  "w" , "sjis" )

        for segms in self.segm_sequences:
            for s in segms:
                for o in s:
                    f.write( o )
                f.write( " | " )
            f.write("\n")
        f.close()

        np.savetxt( os.path.join(dir,"trans.txt") , self.trans_prob , delimiter="\t" )
        np.savetxt( os.path.join(dir,"trans_bos.txt") , self.trans_prob_bos , delimiter="\t" )
        np.savetxt( os.path.join(dir,"trans_eos.txt") , self.trans_prob_eos , delimiter="\t" )


def main():
    segm = HSMMWordSegm( 3 )
    segm.load_data( "data.txt" )
    segm.print_result()

    for it in range(50):
        print( it )
        segm.learn()
        print( "lik =", segm.calc_loglik() )

    segm.learn( True )
    segm.save_result("result")
    segm.print_result()
    return






if __name__ == '__main__':
    main()