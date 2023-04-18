from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib, ssl

smtp_server = "smtp.gmail.com"
smtp_port = 587

me = "1705048.1705100.thesis@gmail.com"
my_password = r"euqoayvlpsbunczg"
you = "farhanakhanhp@gmail.com"

msg = MIMEMultipart('alternative')
msg['From'] = me
msg['To'] = you


def sendmail(object, message):
    try:
        msg['Subject'] = object
        html = '<html><body><p>{0}</p></body></html>'.format(message)
        body = MIMEText(html, 'html')
        
        msg.attach(body)

        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(me, my_password)
            smtp.sendmail(me, you, msg.as_string())
            smtp.quit()
    except Exception as e:
        print('Sending mail failed.\n', e)


def sendmailwithfile(object, message, filename, filelocation):
    '''Send mail with file attachment.
        params: 
            object: subject of the mail
            message: body of the mail
            filename: name of the file to be attached
            filelocation: location of the file to be attached
    '''


    try:
        msg['Subject'] = object
        html = '<html><body><p>{0}</p></body></html>'.format(message)
        part2 = MIMEText(html, 'html')
        msg.attach(part2)

        # Open PDF file in binary mode
        with open(filelocation, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        msg.attach(part)

        text = msg.as_string()

        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(me, my_password)
            server.sendmail(me, you, text)
    except Exception as e:
        print('Sending mail failed.\n', e)
