from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib, ssl

from email.message import EmailMessage


smtp_server = "smtp.gmail.com"
smtp_port = 587

me = "1705048.1705100.thesis@gmail.com"
my_password = r"euqoayvlpsbunczg"
you = "farhanakhanhp@gmail.com"


def sendmail(subject, message):
    try:
        
        msg = EmailMessage()
        msg['From'] = me
        msg['To'] = you

        msg['Subject'] = subject
        msg.set_content(message)

        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(me, my_password)
            smtp.send_message(msg)
    except Exception as e:
        print('Sending mail failed.\n', e)


def sendmailwithfile(subject, message, filename, filelocation):
    '''Send mail with file attachment.
        params: 
            subject: subject of the mail
            message: body of the mail
            filename: name of the file to be attached
            filelocation: location of the file to be attached
    '''

    try:
        
        msg = EmailMessage()
        msg['From'] = me
        msg['To'] = you
        
        msg['Subject'] = subject
        msg.set_content(message)

        # Open file in binary mode
        with open(filelocation, 'rb') as fp:
            file_content = fp.read()
        
        msg.add_attachment(file_content, maintype='application',
                                    subtype='octet-stream', filename=filename)
        
        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(me, my_password)
            smtp.send_message(msg)

    except Exception as e:
        print('Sending mail failed.\n', e)
