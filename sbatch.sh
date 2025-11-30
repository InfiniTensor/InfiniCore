srun --partition=nvidia --nodes=1 --gres=gpu:nvidia:2 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:20:00  \
        --output=output_%j.log          \
        python test/infinicore/ops/bilinear.py --nvidia --verbose --bench --debug
