# ./VocabTree/VocabMatch/VocabMatch vocab.db list.keys list.keys 50 top50_matches.txt
# mcl top50_matches.txt --abc -o clusters.txt
i=1
while IFS= read -r line; do
	mkdir ./clusters/$i;
	for f in $line; do
		echo "$i - $f";
		cp ./dataset/pgm/`ls dataset/pgm/ | awk NR==$f+1` ./clusters/$i/;
	done;
	i=$(($i+1));
done < clusters.txt