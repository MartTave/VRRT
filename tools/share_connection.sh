echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward

sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
sudo iptables -A FORWARD -i enp7s0 -o wlan0 -j ACCEPT
sudo iptables -A FORWARD -i wlan0 -o enp7s0 -m state --state RELATED,ESTABLISHED -j ACCEPT
