$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "..\src"
Push-Location $src
try {
    $datasets = @("Grocery_and_Gourmet_Food", "ML_1MTOPK")
    foreach ($dataset in $datasets) {
        python main.py --model_name BPRMF --dataset $dataset --path ../data --gpu 0 `
            --emb_size 64 --lr 0.001 --l2 0.0001 --batch_size 2048 `
            --epoch 200 --early_stop 5 --random_seed 2026 --num_workers 4 `
            --topk 10,20 --metric NDCG,HR --main_metric NDCG@20 --save_final_results 0

        python main.py --model_name LightGCN --dataset $dataset --path ../data --gpu 0 `
            --emb_size 64 --n_layers 3 --lr 0.001 --l2 0.0001 --batch_size 2048 `
            --epoch 200 --early_stop 5 --random_seed 2026 --num_workers 4 `
            --topk 10,20 --metric NDCG,HR --main_metric NDCG@20 --save_final_results 0

        python main.py --model_name SimGCL --dataset $dataset --path ../data --gpu 0 `
            --emb_size 64 --n_layers 3 --cl_rate 0.2 --temperature 0.2 --eps 0.1 `
            --include_ego 0 --lr 0.001 --l2 0.0001 --batch_size 2048 `
            --epoch 200 --early_stop 5 --random_seed 2026 --num_workers 4 `
            --topk 10,20 --metric NDCG,HR --main_metric NDCG@20 --save_final_results 0
    }
}
finally {
    Pop-Location
}
