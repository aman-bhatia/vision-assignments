name=`basename $1 .pgm`
mkdir ./result/$name
./sift <$1 >./result/$name/$name.key
echo ./result/$name/$name.key > ./result/$name/query_list.key
./VocabTree/VocabMatch/VocabMatch vocab.db list.keys ./result/$name/query_list.key 100 ./result/$name/matches.txt
for f in `cat ./result/$name/matches.txt | awk '{print $2+1}'`; do cp ./dataset/pgm/`ls dataset/pgm/ | awk NR==$f` ./result/$name/; done
