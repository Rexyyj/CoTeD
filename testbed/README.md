# Testbed setup 

CoTeD framework is designed and validated on top of a testbed as the figure below shows, which  consists of two parts: UAV and edge server.  Each part includes three stacks: the hardware stack, the communication software stack, and the application stack. 
![](../figures/testbed.png)

## The hardware stack
For the hardware stack, the devices used in the testbed includes:
* Nvidia Jetson Orin Nano (as the UAV computation unit)
* A laptop featuring an Intel I7-7700HQ CPU and Nvidia GTX 1050Ti GPU (as the edge server) 
* USRP B210 x2 (as the radio interface)

We remark that, since deploying the OAI UE on the ARM based Nvidia Jetson Orin Nano is difficult, an additional laptop is used, acting as the gateway to access the 5G network. The deployment architecture is as below:
```less
    [ Jetson Orin Nano ]
            |
        Ethernet
            |
[ Laptop acting as gateway ]
            |
        [ USRP B210 ]
            |
        5G Network
            |
        [ USRP B210 ]
            |
[ Laptop acting as edge server ]
```

## The communication stack
In the communication software stack, we set up a private 5G connection between the UAV and the edge server by leveraging the widely adopted,  open-source [OperAirInterface (OAI) project](https://openairinterface.org/). 

At the UAV side, we configure the USRP as a 5G user using OAI-UE and connect it to our private 5G gNB. On the edge server side, we create our private  base station by configuring the USRP as a 5G gNB through the OAI-RAN (Radio Access Network) and run the OAI-5GC (Core Network) on the same server. For the configuration details, please refer to the descriptions in the ```openairinterface5G``` submodule in this folder.

In addition, to host edge services, we integrate the OAI-MEC platform to the edge server. This architecture enables  edge services to directly access the 5G network through the UPF function in the OAI-5GC. For the configuration details, please refer to the descriptions in the ```oai-mep``` submodule in this folder.

We also monitor and expose network-level metrics to our framework via the OAI-RNIS. For the configuration details, please refer to the descriptions in the ```oai-rnis``` submodule in this folder.

## The application stack
On the UAV side, we directly run the DNN Head and the CoTeD framework on the Navidia Jetson Orin Nano's Python environment. The following Python packages are needed:
```
numpy
torch
tqdm
requests
pickle
terminaltables
scipy
simplejpeg
pandas
tensorly
```

On the edge server side, we can run the DNN Tail and the CoTeD framework in the container environment or directly on the edge server's Python environment.  
The following Python packages are needed:
```
cherrypy
numpy
torch
pickle
simplejpeg
tensorly
```
