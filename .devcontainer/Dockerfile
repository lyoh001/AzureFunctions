FROM ubuntu:20.04

ENV TZ=Australia/Melbourne

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get install man-db -y \
    && apt-get install wget -y \
    && apt-get install curl -y \
    && apt-get install gnupg -y \
    && apt-get install software-properties-common -y \
    && apt-get install ca-certificates -y \
    && apt-get install apt-transport-https -y \
    && apt-get install lsb-release -y \
    && apt-get install htop -y \
    && apt-get install zsh -y \
    && apt-get install tzdata -y \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add - \
    && apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
    && curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null \
    && AZ_REPO=$(lsb_release -cs) \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | tee /etc/apt/sources.list.d/azure-cli.list \
    && curl -Lo bicep.bin https://github.com/Azure/bicep/releases/latest/download/bicep-linux-x64 \
    && chmod +x ./bicep.bin \
    && mv ./bicep.bin /usr/local/bin/bicep \
    && wget https://packages.microsoft.com/config/debian/10/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get update -y \
    && apt-get dist-upgrade -y \
    && apt-get install nodejs -y \
    && apt-get install awscli -y \
    && apt-get install azure-cli -y \
    && apt-get install dotnet-sdk-3.1 -y \
    && apt-get install git -y \
    && apt-get install python3.8 -y \
    && apt-get install python3-pip -y \
    && apt-get install python3-venv -y \
    && apt-get install terraform -y \
    && npm install -g npm@latest \
    && npm install -g aws-cdk \
    && npm install -g cdktf-cli \
    && npm install -g @azure/static-web-apps-cli \
    && npm install -g azure-functions-core-tools@4 --unsafe-perm true \
    && python3 -m pip install --upgrade pip \
    && sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

EXPOSE 7000