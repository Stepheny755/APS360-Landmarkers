@ECHO OFF
FOR %%L IN (0.005, 0.001, 0.0005) DO (
    FOR %%B IN (32, 16) DO (
        python .\train.py --data_folder datasets --network efficientnet-b0 --freeze_layers False --batch_size %%B --optimizer Adam --learning_rate %%L --num_workers 2
    )
)