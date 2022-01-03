sudo apt update
sudo apt install ubuntu-desktop -y
sudo apt-get install tigervnc-standalone-server -y
sudo apt install gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal -y

vncserver :1
vim ~/.vnc/xstartup

# https://ubuntu.com/tutorials/ubuntu-desktop-aws#3-configuring-the-vnc-server

