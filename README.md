# Lightweight Remote Sensing Change Detection with Progressive Aggregation and Supervised Attention

## Get Start
**1. Download Datasets**

- [LEVIR](https://justchenhao.github.io/LEVIR/) 

- [BCDD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

- [SYSU](https://github.com/liumency/SYSU-CD)

Prepare datasets into following structure,
```
├─Train
    ├─A
    ├─B
    ├─label
    └─list
├─Val
    ├─A
    ├─B
    ├─label
    └─list
├─Test
    ├─A
    ├─B
    ├─label
    └─list
```

**2. Train**
```
sh ./tools/train.sh
```

**3. Test**
```
sh ./tools/test.sh
```

### Acknowlogdement

This repository is built under the help of the projects [BIT_CD](https://github.com/justchenhao/BIT_CD), 
[CDLab](https://github.com/Bobholamovic/CDLab), and [MobileSal](https://github.com/yuhuan-wu/MobileSal) for academic use only.
