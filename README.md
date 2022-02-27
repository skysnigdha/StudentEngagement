# StudentEngagement 
Sequence of Run:

python3.9 SE_cuttingVideo.py -v videoData/Debasree_20211221/2021-12-21\ 15-11-00-Debasree.mp4 -s 00:00:06 -d 00:16:16

python3 SE_lookingAtScreen.py -d 20211221 

python3 SE_backgroundSubtractor_v2.py -d 20211221 > 20211221_ForegroundExtractor_1.txt

python3 SE_compareInstructorAudience_v3.py -d 2021-12-21 -i Soumyajit > 20211221__InstructorAudienceFacialMatch_v3.txt

python3.9 SE_gazeDetection_v7.py -d 2021-12-21 -o gazeRatioTry

python3 SE_gazeAnalysis_v4.py -d 2021-12-21 -fd gazeRatioTry/ -i Soumyajit > 20211221_gazeAnalysis_v4.txt

python3 SE_gazeAnalysis_v5.py -d 2021-12-21 -fd gazeRatioTry/ -i Soumyajit > 20211221_gazeAnalysis_v5.txt

_____________________________________________________________________________________

python3.9 SE_gazeDetection_v7.py -d 2021-12-21 -o gazeRatioFull -ff _InstructorAudienceFacialMatch_v3_Full.txt


____________________________________________________________________________________

python3 SE_lookingAtScreenPerformance.py -d 2021-12-21 -i Soumyajit > 20211221_lookingAtScreenOnlyPerformance.txt

_____________________________________________________________________________________
InWildExperiments

python3 SE_lookingAtScreen.py -d 20210706 

python3 SE_backgroundSubtractor_v2.py -d 20210706 > 20210706_ForegroundExtractor.txt

python3 SE_compareInstructorAudience_v3.py -d 2021-07-06 -i Snigdha > 20210706_InstructorAudienceFacialMatch_v3.txt

python3.9 SE_gazeDetection_v7.py -d 2021-07-06 -o gazeRatioTry

python3 SE_gazeAnalysis_v4.py -d 2021-07-06 -fd gazeRatioTry/ -i Snigdha > 20210706_gazeAnalysis_v4.txt

python3 SE_gazeAnalysis_v5.py -d 2021-07-06 -fd gazeRatioTry/ -i Snigdha > 20210706_gazeAnalysis_v5.txt

20210706 Snigdha
20210803 Pragma
20210810 SC
20210817 Bishakh
20210825 Utkalika

_______________________________________________________________________________________

python3 SE_lookingAtScreen.py -d 20210816 -o LookingAtScreen/AOS/ -v videoData/AOS/

python3 SE_backgroundSubtractor_v2.py -d 20210816 -v videoData/AOS/ > 20210816_ForegroundExtractor.txt

python3 SE_compareInstructorAudience_v3.py -d 2021-08-16 -v videoData/AOS/ -fd LookingAtScreen/AOS/ -i Snigdha > 20210816_InstructorAudienceFacialMatch_v3.txt

python3.9 SE_gazeDetection_v7.py -d 2021-08-16 -v videoData/AOS/ -o gazeRatioTry/AOS

python3 SE_gazeAnalysis_v4.py -d 2021-08-16 -fd gazeRatioTry/AOS/ -i Snigdha > 20210816_gazeAnalysis_v4.txt

python3 SE_gazeAnalysis_v5.py -d 2021-08-16 -fd gazeRatioTry/AOS/ -i Snigdha > 20210816_gazeAnalysis_v5.txt


AOS
20210816 SC
20210817 SC
20210823 SC
20210824 SC

Blockchain 
20210819 SC

ReadingGroup
20210819 Soumi
20210826 Paheli
