# Deep Learning for Computer Vision practical course WS 2016/17
# Rajat Jain
# Protein function prediction from 2D representation of 3D structure

import datetime
import smtplib
from email.mime.text import MIMEText

'''
Returns a timestamped render of the filename.
eg. filename = train.log
output = 2017-03-01-11-11-12_train.log

Useful to keep the logs organised.
'''
NOW__STRFTIME = datetime.datetime.now()


def time_stamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return NOW__STRFTIME.strftime(fmt).format(fname=fname)


# Change these addresses accordingly
from_address = '<from>@gmail.com'
to_address = '<to>'


def send_mail(body, subject='Training Report'):
    # Create a text/plain message
    msg = MIMEText(body)

    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = to_address

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(from_address, "<password>")
        server.sendmail(from_address, [to_address], msg.as_string())
        server.quit()
    except:
        pass
