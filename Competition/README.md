# IR for CLEF E Health


## Prep
Create the enviorment. 
```bash
conda env create -f environment.yaml
```
Download the data using the following command
```bash
wget -c https://www.dropbox.com/s/ixnqt33u5xeelth/clef2018collection.tar.gz
tar -xvf clef2018collection.tar.gz
sh build_index.sh
```
If everything went as expected you should get should have a functional index

```bash
wget -c https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/assessments/Final%20assessment/CLEF2018_qread_20180914.txt
wget -c https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/assessments/Final%20assessment/CLEF2018_qtrust_20180914.txt
wget -c https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/assessments/Final%20assessment/CLEF2018_qrel_20180914.txt
'''
## Overall Structure
PYANSWERINI
##
