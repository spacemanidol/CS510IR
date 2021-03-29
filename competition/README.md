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
sha256sum clef2018collection
```
If everything went as expected you should get ####

```bash
wget -c https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/assessments/Final%20assessment/CLEF2018_qread_20180914.txt
wget -c https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/assessments/Final%20assessment/CLEF2018_qtrust_20180914.txt
wget -c https://github.com/CLEFeHealth/CLEFeHealth2018IRtask/blob/master/assessments/Final%20assessment/CLEF2018_qrel_20180914.txt
'''
## Overall Structure
PYANSWERINI -> BERT Model

##