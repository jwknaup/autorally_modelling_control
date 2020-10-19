map_parameters_ = ...
	 	{
	 		{
				{-8.62, 8.38, 2.36, 0, 5.9821, 0},					 
				{-16.94, -0.19, -0.80, 5.9821, 18.7621, 0.1674}, 
				{-8.81, -8.64, -0.80, 24.7552, 11.726, 0}, 
				{-0.12, 0.05, 2.36, 36.4702, 19.304, 0.1627}, 
				{-4.37, 4.17, 2.36, 55.774, 5.919, 0}
			},

			{
				{2.78,-2.97,-0.6613, 0, 3.8022, 0},
				{10.04,6.19, 2.4829, 3.8022, 18.3537,0.1712 }, 
				{1.46, 13.11,2.4829, 22.1559, 11.0228 , 0},
	 			{-5.92, 3.80, -0.6613, 33.1787 , 18.6666, 0.1683},
				{-0.24, -0.66,-0.6613, 51.8453, 7.2218, 0}
			}
	 	};

%%
master = ros.Core;
node = ros.Node('carsim2');
pub = ros.Publisher(node,'/state','std_msgs/Float32MultiArray');
sub = ros.Subscriber(node,'/state','std_msgs/Float32MultiArray');

for ii = 1:5
%     state_msg = rosmessage('std_msgs/Float32MultiArray');
%     state_msg.Data = [1.0, 2.0];
%     send(pub, state_msg);
% %     sub.LatestMessage
    msg = receive(sub)
end

clear('pub','sub','node')
clear('master')
