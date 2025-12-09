# --- Job Submission ---
bsub -J "test" \
     -R "select[hname!='e230-dgx1-1' && hname!='e230-dgx2-1' && hname!='e230-dgx2-2']" \
     -gpu "num=1:j_exclusive=yes:gmem=24GB" \
     -q gpu \
     -L /bin/bash "bash -i -c '
        . /home/s124z/env_scripts/yolo_init.sh &&
        cd /home/s124z/heico_code &&
        python main.py 

    '"

bsub -J "deepsort_100_epochs" \
     -R "select[hname!='e230-dgx1-1' && hname!='e230-dgx2-1' && hname!='e230-dgx2-2']" \
     -gpu "num=1:j_exclusive=yes:gmem=24GB" \
     -q gpu \
     -L /bin/bash "bash -i -c '
        . /home/s124z/env_scripts/yolo_init.sh &&
        cd /home/s124z/YOLO &&
        python train.py 

    '"

bsub -J "kfold" \
     -R "select[hname!='e230-dgx1-1' && hname!='e230-dgx2-1' && hname!='e230-dgx2-2']" \
     -gpu "num=1:j_exclusive=yes:gmem=24GB" \
     -q gpu \
     -L /bin/bash "bash -i -c '
        . /home/s124z/env_scripts/yolo_init.sh &&
        cd /home/s124z/heico_code &&
        python kfold.py 

    '"    



bsub -J "yolo_heico_aug_original" \
     -R "select[hname!='e230-dgx1-1' && hname!='e230-dgx2-1' && hname!='e230-dgx2-2']" \
     -gpu "num=1:j_exclusive=yes:gmem=24GB" \
     -q gpu \
     -L /bin/bash "bash -i -c '
        . /home/s124z/env_scripts/yolo_init.sh &&
        cd /home/s124z/heico_code &&
        python withaug.py 

    '"



bsub -J "yolo_heico_aug" \
     -R "select[hname!='e230-dgx1-1' && hname!='e230-dgx2-1' && hname!='e230-dgx2-2']" \
     -gpu "num=1:j_exclusive=yes:gmem=24GB" \
     -q gpu \
     -L /bin/bash "bash -i -c '
        . /home/s124z/env_scripts/yolo_init.sh &&
        cd /home/s124z/heico_code &&
        python test_as_train.py 

    '"    