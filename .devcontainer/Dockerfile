FROM ubuntu:20.04

ENV TZ=Australia/Melbourne

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get install wget -y \
    && apt-get install curl -y \
    && apt-get install htop -y \
    && apt-get install zsh -y \
    && apt-get install tzdata -y \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && apt-get update -y \
    && apt-get dist-upgrade -y \
    && apt-get install dotnet-sdk-6.0 -y \
    && apt-get install git -y \
    && apt-get install nodejs -y \
    && apt-get install python3.8 -y \
    && apt-get install python3-pip -y \
    && apt-get install python3-venv -y \
    && npm install -g npm@latest \
    && npm install -g azure-functions-core-tools@4 --unsafe-perm true \
    && sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

EXPOSE 7000