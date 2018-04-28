#!/bin/bash

apt-get update

sudo apt-get install figlet -y

apt-get install squid3 -y

echo "[ ]====================================[ ]";
echo "[ ]      PRIMEIRO COMANDO DA NET       [ ]";
echo "[ ]     Squid install by JoeLinux      [ ]";
echo "[ ]====================================[ ]";

sleep 2s

echo "-------|--------|--------|--------|-------"
echo "-------|--------|--------|--------|-------"
echo "-------|--------|--------|--------|-------"
echo "-------|--------|--------|--------|-------"
echo "-------|--------|--------|--------|-------"
echo "-------|--------|--------|--------|-------"

echo "[ ]====================================[ ]";
echo "[ ]              Ubuntu 14             [ ]";
echo "[ ]     Squid install by JoeLinux      [ ]";
echo "[ ]====================================[ ]";

sleep 2s


echo "Port 443" >> /etc/ssh/sshd_config 

echo "Vou apagar o arquivo squid.conf e criar um novo"

sleep 1s

cd /etc/squid3/

echo Add o ip do host nos arquivos accept e squid.conf

read ip

echo "$ip">> accept
echo ".com.br">> accept
echo "vivo">> accept
echo "claro">> accept
echo "tim">> accept
echo "oi">> accept
echo "tigo">> accept

chmod 777 squid.conf

rm squid.conf

echo "http_port 80" >> squid.conf
echo "http_port 8080" >> squid.conf
echo "http_port 3128" >> squid.conf
echo "visible_hostname joelinux" >> squid.conf
echo "acl accept src $ip" >> squid.conf
echo 'acl br url_regex -i "/etc/squid/accept"' >> squid.conf
echo 'acl br url_regex -i "/etc/squid3/accept"' >> squid.conf
echo "acl all src 0.0.0.0/0.0.0.0" >> squid.conf
echo "http_access allow accept" >> squid.conf
echo "http_access allow br" >> squid.conf
echo "http_access deny all" >> squid.conf

echo "http_port 80" >> squid3.conf
echo "http_port 8080" >> squid3.conf
echo "http_port 3128" >> squid3.conf
echo "visible_hostname joelinux" >> squid3.conf
echo "acl accept src $ip" >> squid3.conf
echo 'acl br url_regex -i "/etc/squid/accept"' >> squid3.conf
echo 'acl br url_regex -i "/etc/squid3/accept"' >> squid3.conf
echo "acl all src 0.0.0.0/0.0.0.0" >> squid3.conf
echo "http_access allow accept" >> squid3.conf
echo "http_access allow br" >> squid3.conf
echo "http_access deny all" >> squid3.conf

echo "aguarde um pouco que vamos configurar para você"

sleep 1s

service ssh restart

service squid3 restart

echo "Powered by Primeiro Comando" 
echo "~Coded by JoeLinux~" 
echo "Para adicionar um usuário:"
echo "useradd -M -s /bin/false nomeuser" 
echo "Para mudar a senha:"
echo "passwd  nomeuser"

sleep 2s 

banner=$ figlet Obrigado
echo "aproveite sua VPS :)"

cd

rm squid.sh

cd

rm ubuntu14.sh
