path="data/datasets/ImageNet/imagenet_rand118/"
mkdir -p $path
cd $path

for file in "imagenet_118.txt" "imagenet_30_A.txt" "imagenet_30_B.txt" "imagenet_30_C.txt"; do
    wget http://www.robots.ox.ac.uk/~vgg/research/DTC/data/datasets/ImageNet/imagenet_rand118/${file}
done

cd ../
