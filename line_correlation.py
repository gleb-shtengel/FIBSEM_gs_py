def line_correlation(stack, center, step, nsteps, line_axis, correlation_axis):
    '''
    Computes line-to-line normalized zero-mean correlation within 3D stack. gleb.shtengel@gmail.com
    
    Parameters:
        stack : 3D array
            vEM data stack
        center : list or array of three coordintes - through which the reference correlation line goes
        step : int 
            # of pixel in a single shift step
        nsteps : int
            # of offset steps to compute correlation
        line_axis : int
            axis of the line to be correlated. 0=z, 1=y, 2=x
        correlation_axis : int
            axis of the correlation displacement. 0=z, 1=y, 2=x
    
    Returns:
        ncc : list of floats
        normalized zero-mean cross-correlation
    '''
    cz, cy, cx = center
    ncc = []
    ir = 3 - line_axis - correlation_axis  # "remaining" axis
    c_corr = center[correlation_axis]
    c_rem = center[ir]
    
    '''
    original axes: 
    0, 1, 2
    
    Axes for correlation:
    corr_direction, line_direction, remaining_direction
    
    line_axis = X ( or 2)
    correlation_axis = Y (or 1)
    should be 1, 2, 0
    0, 1, 2 -> np.swapaxes(stack, correlation_axis, 0) -> 1, 0, 2
    1, 0, 2 -> np.swapaxes(stack, line_axis, 1) -> 1, 2, 0  - correct
    
    line_axis = X ( or 2)
    correlation_axis = Z (or 0)
    should be 0, 2, 1
    0, 1, 2 -> np.swapaxes(stack, correlation_axis, 0) -> 0, 1, 2
    0, 1, 2 -> np.swapaxes(stack, line_axis, 1) -> 0, 2, 1  - correct
    
    line_axis = Y ( or 1)
    correlation_axis = Z (or 0)
    should be 0, 1, 2
    0, 1, 2 -> np.swapaxes(stack, line_axis, 1) -> 0, 1, 2  - correct
    
    line_axis = Y ( or 1)
    correlation_axis = X (or 2)
    should be 2, 1, 0
    0, 1, 2 -> np.swapaxes(stack, correlation_axis, 0) -> 2, 1, 0
    2, 1, 0 -> np.swapaxes(stack, line_axis, 1) -> 2, 1, 0  - correct
    
    line_axis = Z ( or 0)
    correlation_axis = X (or 2)
    should be 2, 0, 1
    0, 1, 2 -> np.swapaxes(stack, line_axis, 1) -> 1, 0, 2
    1, 0, 2 -> np.swapaxes(stack, correlation_axis, 0) -> 2, 0, 1 - correct
    
    line_axis = Z ( or 0)
    correlation_axis = Y (or 1)
    should be 1, 0, 2
    0, 1, 2 -> np.swapaxes(stack, line_axis, 1) -> 1, 0, 2
    1, 0, 2 -> np.swapaxes(stack, correlation_axis, 0) -> 1, 0, 2 - correct
    
    '''
    
    if (line_axis + correlation_axis) == 1: #If one of the is 1 and the other is 0, only need to swap once
        new_stack = np.swapaxes(stack, line_axis, 1)
    else:
        if line_axis == 0:
            new_stack = np.swapaxes(np.swapaxes(stack, line_axis, 1), correlation_axis, 0)
        else:
            new_stack = np.swapaxes(np.swapaxes(stack, correlation_axis, 0), line_axis, 1)
    #now the line axis is axis 1, and correlation axis is axis0
    #print('remaining_axis={:d}, coordinate={:d}, correlation_start_coord={:d}'.format(ir, c_rem, c_corr))
    #print(new_stack.shape)
    
    line0 = np.squeeze(new_stack[c_corr, :, c_rem])
    line0 = line0 - line0.mean()
    cs = np.arange(c_corr, c_corr+step*nsteps, step)
    for c in cs:
        line1 = np.squeeze(new_stack[c, :, c_rem])
        line1 = line1 - line1.mean()
        cc00 = np.sum(line0*line0)
        cc11 = np.sum(line1*line1)
        cc10 = np.sum(line0*line1)
        ncc.append(cc10 / np.sqrt(cc00*cc11))

    return ncc