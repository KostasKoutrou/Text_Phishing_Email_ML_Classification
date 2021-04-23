import mailbox
import email
import os
import glob
import shutil
import random


def movefiles(mdirpath):
    folders = []
    for root, dirs, files in os.walk(mdirpath):
        if files:
            folders.append(root)
    
    for folder in folders:
        print("Processing " + folder)
        #The 2 next command are commented because the program was already
        #executed once so the directories are already created.
        try:
            os.makedirs(folder + "\\cur")
        except:
            pass
        try:
            os.makedirs(folder + "\\new")
        except:
            pass
        for file in glob.glob(folder + "\\[0-9]*"):
            shutil.move(file, folder + "\\cur")
    return



def maildir2mailbox(maildirname, mboxfilename):
    global emailsindex
    global maxemails
    #open the existing maildir and the target mbox file
#    maildir = mailbox.Maildir(maildirname, email.message_from_file)
    maildir = mailbox.Maildir(maildirname)
    mbox = mailbox.mbox(mboxfilename)

    # lock the mbox
    mbox.lock()

    # iterate over messages in the maildir and add to the mbox
    for msg in maildir:
        if emailsindex < maxemails:
            mbox.add(msg)
            emailsindex += 1
        else:
            print("wrote " + str(maxemails) + " emails")
#            break
            return

    # close and unlock
    mbox.close()
    maildir.close()



mdirpath = "D:\\maildir"
mboxpath = "D:\\enronmbox"

folders = []
for root, dirs, files in os.walk(mdirpath):
    if files:
        folders.append(root)

random.shuffle(folders)

maxemails = 5000
fileindex = 0
emailsindex = 0
for folder in folders:
    if(fileindex < 9):
        folder = folder.replace("\\cur", "")
        writepath = mboxpath + "\\enron_" + str(fileindex) + "_" + str(maxemails) + ".mbox"
        if(emailsindex < maxemails):
            print("Writing " + folder + " -> " + writepath)
            maildir2mailbox(folder, writepath)
        else:
            emailsindex = 0
            fileindex += 1