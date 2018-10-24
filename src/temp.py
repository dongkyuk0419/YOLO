x = convblock2(64,x)
x = convblock2(128,x)
x = convblock3(256,x)
passthrough = x
x = MaxPooling2D(strides=2)(x)
x = convblock3(512,x)
for i in range(0,2):
	x = convconv(1024,3,x)
passthrough = convconv(64,1,passthrough)
passthrough = Lambda(reorg)(passthrough)
x = concatenate([x,passthrough])
x = convconv(1024,3,x)
y = Conv2D(out,1,padding = 'same')(x)
