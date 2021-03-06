# Steps to deploy the bot

1. Create an instance on AWS EC2 Free tier with appropriate configurations
2. Update the packages and the package repository by using ```sudo apt update && sudo apt upgrade```
3. Install python by using ```sudo apt install python3-pip```
4. Configure your git credentials 
5. Clone the repository
6. Move into that folder 
7. Install virtualenv package if you do not have it by using ```sudo pip3 install virtualenv```
8. Install dependencies for opencv by using ```sudo apt install python3-opencv```
9. Create a virtualenv by using ```virtualenv env```
10. Activate the virtual environment by using ```source env/bin/activate```
11. Install all the dependencies by using ```pip3 install -r requirements.txt```
12. Configure screen by using ```screen```
13. Run the bot by using ```python3 bot.py```
14. Detach the screen by using ```Ctrl + A + D```
15. Create a monthly cron job to clear the uploads by using ```0 0 1 * * cd HateDetectronn && /usr/bin/python3 clearUploads.py```