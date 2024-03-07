# Setting up WSL

There are numerous guides to setting up WSL, and this is not a comprehensive guide. Instead this might help you setup the basics.

#### If you are using Docker Desktop 

To enable **Developing inside a Container** you must configure the integration between Docker Desktop and WSL on your machine.

>1. Launch Docker Desktop
>2. Open **Settings > General**. Make sure the *Use the WSL 2 based engine" is enabled.
>3. Navigate to **Settings > Resources > WSL INTEGRATION**.
>      - Ensure *Enable Integration with my default WSL distro" is enabled.
>      - Enable the Ubuntu-18.04 option.
>4. Select **Apply & Restart**

## Configure Git in Ubuntu WSL environment

The next step is to configure Git for your Ubuntu WSL environment. We will use the bash prompt from the previous step to issue the following commands:

Set Git User Name and Email

``` bash
git config --global user.name "Your Name"
git config --global user.email "youremail@yourdomain.com"
```

Set Git [UseHttps](https://github.com/microsoft/Git-Credential-Manager-Core/blob/main/docs/configuration.md#credentialusehttppath)

``` bash
git config --global credential.useHttpPath true
```

Configure Git to use the Windows Host Credential Manager

``` bash
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-manager-core.exe"
```

## Install Azure CLI On WSL

In your Ubuntu 18.04(WSL) terminal from the previous step, follow the directions [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-linux) to install Azure CLI.


Install Azure CLI and authorize:
```bash
az login
az account set  --subscription="<your_subscription_guid>"
az account show
```
