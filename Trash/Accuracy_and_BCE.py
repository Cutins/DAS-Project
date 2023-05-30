###############################################################################
# Cost Function
def cost_fucnt(Y,xT, mask=None):
    xT0 = xT[0] #It's a scalar

    if CostFunct == "BinaryCrossEntropy":
            J = -(Y*np.log(xT0) + 1e-10)-((1-Y)*(np.log(1-xT0)) + 1e-10)
            dJ = -Y/(xT0 + 1e-10) + (1-Y)/(1-xT0 + 1e-10)


    if CostFunct == "Quadratic":
            J = (xT0 - Y)*(xT0 - Y)
            dJ = 2*(xT0 - Y)

         
    if mask is not None:
            dJ = dJ*mask

    return J, dJ

###############################################################################
def accuracy(xT,Y):
    error = 0
    success = 0

    if xT>0.5:
        if Y==0: # Missclassified
            error += 1
        else:
            success += 1
    else:
        if Y==0:  #Correctly classified
            success += 1
        else:
            error +=1 

    return success, error




# GO!
# for k in range(EPOCHS):
#     if k%10==0 and k!=0:
#         print(f'Cost at k={k:d} is {J[k-1]:.4f}')

#     for batch_n in range(N_BATCH):
#         delta_u = 0
#         for batch_sample in range(BATCHSIZE):
#             idx = (batch_n*BATCHSIZE) + batch_sample
#             if idx >= len(df_train):
#                 continue
            
#             xx[batch_sample] = forward_pass(INPUTS[idx], uu)

#             loss, llambdaT = cost_fucnt(LABEL[idx], xx[batch_sample,-1,:], mask)
#             J[k] += loss / len(df_train)
#             _, Grad = backward_pass(xx[batch_sample], uu, llambdaT)

#             delta_u += Grad / BATCHSIZE
        
#         uu = uu - STEPSIZE*delta_u 
#         NormGradientJ[k] += np.linalg.norm(delta_u) / N_BATCH
    
    if k == EPOCHS-1:
        for img in range(BATCHSIZE):
            print(f"Label for Image {img} was {LABEL[img]} but is classified as:", xx[img,-1, 0])
            success, error = accuracy(xx[img,-1, 0],LABEL[img])
            Success += success
            Error += error

        PercentageOfSuccess = (Success/(Success+Error))*100
        print("Correctly classified point: ", Success)
        print("Wrong classified point: ", Error)
        print("Percentage of Success: ", PercentageOfSuccess)