for f in `ls ./dataset/pgm/`;
	do `./sift <./dataset/pgm/$f >./dataset/key_files/$f.key`;
done;