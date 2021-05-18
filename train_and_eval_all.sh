echo "####### exp 1 #######"
python3 -m cr_cw1.run.train_and_eval --model="base_cnn" --epoch=20 --batch=20 --patience=3 --normalise_data
python3 -m cr_cw1.run.train_and_eval --model="two_cnn" --epoch=20 --batch=20 --patience=3 --normalise_data
python3 -m cr_cw1.run.train_and_eval --model="three_cnn" --epoch=20 --batch=20 --patience=3 --normalise_data


echo "\n###### exp 2 ########"
python3 -m cr_cw1.run.train_and_eval --model="reg_base_cnn" --epoch=20 --batch=20 --patience=3 --normalise_data
python3 -m cr_cw1.run.train_and_eval --model="reg_two_cnn" --epoch=20 --batch=20 --patience=3  --normalise_data
python3 -m cr_cw1.run.train_and_eval --model="reg_three_cnn" --epoch=20 --batch=20 --patience=3  --normalise_data
