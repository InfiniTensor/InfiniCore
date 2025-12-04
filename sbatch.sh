srun --partition=mt --nodes=1 --gres=gpu:mt:2 --ntasks=1 --cpus-per-task=16 --mem=256G --time=00:01:00  \
        --output=output_%j.log          \
        python test/infinicore/ops/bilinear.py --moore --verbose --bench --debug
