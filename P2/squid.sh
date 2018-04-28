#!/bin/bash

echo "[ ]====================================[ ]";
echo "[ ]      PRIMEIRO COMANDO DA NET       [ ]";
echo "[ ]     Squid install by JoeLinux      [ ]";
echo "[ ]====================================[ ]";
echo "";
echo "[!] squid.sh irá realizar modificações, permitir? [S/n]" ; 
read resposta
if [ $resposta == "s" ] ; 
  then
    echo " "
  else
    exit
fi

echo "[!] Digite o nome correspondente à distribuição do droplet";
echo "----------------------------------------------------------"
echo "ubuntu14";
echo "ubuntu16";
echo "debian8";
echo "----------------------------------------------------------"
echo "Disponíveis somente para membros pro"
echo "centos7";
echo "fedora24";
echo "----------------------------------------------------------"
echo "Caso o seu não corresponda às opções, diga ao mestre Joe"
sleep 2s
echo "Se não souber a distribuição, só lamento"
sleep 2s

read droplet

wget joelinux.hol.es/$droplet.sh

bash $droplet.sh
