'''
File: /sendEmail.py
Created Date: Saturday November 4th 2023
Author: Zihan
-----
Last Modified: Saturday, 4th November 2023 9:42:29 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
4-11-2023		Zihan	Init
'''

import os
import smtplib
from email.mime.text import MIMEText
import time

class Email:
    '''
    inputable args:
    - content
    - subject
    - from
    - to
    - task_id
    '''
    def __init__(self, **kwargs):
        # Set up the SMTP server
        self.smtp = smtplib.SMTP('smtp.elasticemail.com', 2525)
        self.smtp.login('wzh4464@gmail.com',
                        '4A83AA86DD7F2E9EB8A3C8CE47599BCEE396')

        # if content is given, use it
        if 'content' in kwargs:
            self.msg = MIMEText(kwargs['content'])
        else:
            self.msg = MIMEText('Program finished')

        if 'subject' in kwargs:
            self.msg['Subject'] = kwargs['subject']
        else:
            self.msg['Subject'] = 'Program finished'

        if 'from' in kwargs:
            self.msg['From'] = kwargs['from']
        else:
            self.msg['From'] = 'wzh4464@gmail.com'

        if 'to' in kwargs:
            self.msg['To'] = kwargs['to']
        else:
            self.msg['To'] = 'zihan.wu@my.cityu.edu.hk'

        # self.msg['Subject'] = 'Program finished'
        # self.msg['From'] = 'wzh4464@gmail.com'
        # self.msg['To'] = 'zihan.wu@my.cityu.edu.hk'
        # if given task id, add it to the subject
        if 'task_id' in kwargs:
            self.msg['Subject'] = self.msg['Subject'] + \
                ' ' + str(kwargs['task_id'])
            self.task_id = kwargs['task_id']
        else:
            # set task_id to Current Time
            self.task_id = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # # make a `.email_sent` file to tmp in system
        # self.sent_file = '/tmp/' + self.task_id + '.email_sent'
        # # f = open(self.sent_file, 'w')
        # # f.write('unsend')
        # with open(self.sent_file, 'w') as f:
        #     f.write('unsend')
            
    def setContent(self, content):
        self.msg.set_payload(content)
        
    def setSubject(self, subject):
        self.msg.replace_header('Subject', subject)

    def send(self):
        # print(self.msg)
        self.smtp.send_message(self.msg)
        print('Email sent')
        # if os.path.isfile(self.sent_file):
        #     content = open(self.sent_file, 'r').read()
        #     if content == 'unsend':
        #         self.smtp.send_message(self.msg)
        #         print('Email sent')
        #         # f = open(self.sent_file, 'w')
        #         # f.write('sent')
        #         with open(self.sent_file, 'w') as f:
        #             f.write('sent')
        #         self.smtp.quit()
        #     elif content == 'sent':
        #         print('Already sent')
        #     else:
        #         raise Exception('Invalid content in .email_sent file')
        # else:
        #     raise Exception('No .email_sent file')

if __name__ == "__main__":
    pass
    # # Count the number of files in the directory
    # num_files = len([f for f in os.listdir('/home/zihan/denElli/result/timebased_2')
    #                 if os.path.isfile(os.path.join('/home/zihan/denElli/result/timebased_2', f))])

    # # if already sent, exit
    # if os.path.isfile('/home/zihan/denElli/result/timebased_2/sent__.txt'):
    #     print('Already sent')
    #     exit()

    # # Check if the number of files exceeds 6000
    # if num_files > 6000:
    #     # Set up the SMTP server
    #     smtp = smtplib.SMTP('smtp.elasticemail.com', 2525)
    #     smtp.login('wzh4464@gmail.com', '4A83AA86DD7F2E9EB8A3C8CE47599BCEE396')

    #     # Create the email
    #     msg = MIMEText(
    #         'The number of files in result/timebased_2 is ' + str(num_files))
    #     msg['Subject'] = 'File Count Alert'
    #     msg['From'] = 'wzh4464@gmail.com'
    #     msg['To'] = 'zihan.wu@my.cityu.edu.hk'

    #     # create a file to record as sent
    #     f = open('/home/zihan/denElli/result/timebased_2/sent__.txt', 'w')
    #     f.write('sent')

    #     # Send the email
    #     smtp.send_message(msg)
    #     print('Email sent')
    #     smtp.quit()