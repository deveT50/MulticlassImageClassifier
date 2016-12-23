# 多クラス画像分類器  
chainerのexampleを改造したもの  
画像の多クラス分類を実行する  
環境：ubuntu14.04, python2.7, chainer1.14.0  

こちらの方々のソースコードを更に改変したものです：  
<http://d.hatena.ne.jp/shi3z/20150709/1436397615>  
<http://hi-king.hatenablog.com/entry/2015/06/11/021144>  

##使い方  
下記の手順を実行して下さい。  


1ディレクトリ中にクラスの名前を持つフォルダが複数入ったようなデータセットを用意して下さい。  
それぞれのクラスディレクトリの画像は手動で分類されているものとします。  
（このように）  


データセット  
|  
|ークラス１
|　　|ー画像１   
|   ー画像２  
|  
|ークラス２  


このファイル（README.md）と同じ階層で、  
  
'$ python 01_make_train_data.py データセットフォルダ名'  
データセット中の画像が１つのディレクトリ「images」にコピーされ、画像はリネームされます。  
また、train.txt, test.txt, label.txt の３つのファイルが作成されます。  


'$ python 02_crop.py images cropedImages'  
「images」フォルダに格納された画像を256*256にリサイズし、新たなディレクトリ「cropedImages」にコピーします。  


ここで、test.txtとtrain.txtのパスを手動（ctrl+h）で修正します。  
images　すべてを　cropedImages　に  


'$python 03_compute_mean.py train.txt'  
正規化のための平均画像「mean.npy」が作成されます。  


'$python 04_train_imagenet_color.py -B 8 -g 0 -E 50 train.txt test.txt'  
学習プログラムです。  
-b はミニバッチあたりの枚数です。  
-g 0　はGPU（ID=0）を使用することを意味します。-1がCPUです。  
-e はエポック数です。  
modelhdf5 が出力されます。ニューラルネットの重みとバイアスを保存したファイルです。  
sigma.npy が作成されます。このファイルは平均０，分散１に正規化するための標準偏差（シグマ）を保存しています。  

##フォルダ構成  
###学習用
* 01_make_train_data.py ・・・・・・・・・・・・・・ train/test/label.txtを作成する  
* 02_crop.py ・・・・・・・・・・・・・・・・・・・・・・・・・ 画像を２５５＊２５５に変形する  
* 03_compute_mean.py ・・・・・・・・・・・・・・・・・ 平均画像を計算して出力する  
* 04_train_imagenet_color.py ・・・・・・・・・ ニューラルネットを訓練してパラメータファイルを出力する  
* 05_test.py ・・・・・・・・・・・・・・・・・・・・・・・・・ （未完成）  
* 06_predict.py ・・・・・・・・・・・・・・・・・・・・・・ （未完成）  
* network.py ・・・・・・・・・・・・・・・・・・・・・・・・・ ニューラルネットの構造定義  
* mean.npy ・・・・・・・・・・・・・・・・・・・・・・・・・・・ 平均画像ファイル  
* modelhdf5 ・・・・・・・・・・・・・・・・・・・・・・・・・・ 訓練済みニューラルネットのパラメータファイル  
* sigma.npy ・・・・・・・・・・・・・・・・・・・・・・・・・・ 標準偏差ファイル  


###ユーティリティ  
utilityフォルダ内  
  
* 00_increase_dataset.py ・・・・・・・・・・・・・ 画像にノイズを付加してデータを増やす[こちらを使用](http://qiita.com/bohemian916/items/9630661cd5292240f8c7 ""）  
'$python 00_increase_dataset.py データセットフォルダ名'  
trans_imagesフォルダが作成されます  
  
以下ファイルの使用方法は学習用の対応する番号と同様  
* 01_image2gray.py ・・・・・・・・・・・・・・・・・・・ カラー画像を白黒にする  
* 02_crop_64.py ・・・・・・・・・・・・・・・・・・・・・・ 画像を64*64に変形する  
* 03_compute_mean_mono.py ・・・・・・・・・・・・ 平均画像を計算する(モノクロ画像用)  




