from win10toast import ToastNotifier

def balloon_tip(title, msg):
    try:
        toaster = ToastNotifier()
        toaster.show_toast(title, msg, duration=6)
    except Exception as e:
        print(e)