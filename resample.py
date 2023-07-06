import numpy as np

def resample(data_in, n_points, tot_steps=10000, tolerance=100):

    # * data is a list of m points, each point of dimemsion k
    #   - i.e.:
    #
    #     data = [[c_00,c_11,...,c_0k], 
    #             [c_10,c_11,...,c_1k], 
    #             ...,                 
    #             [c_m0,c_m1,...,c_mk], 
    #
    #   - e.g. 3 points of dimension 2:
    #
    #     data = [[x_0,y_0],
    #             [x_1,y_1],
    #             [x_2,y_2]]
    #
    # * n_points is the number of resampled points
    #
    
    assert len(data_in) > 0
    
    all_segment_vec = []
    #print("data in", data_in)
    for i in range(1,len(data_in)):
        all_segment_vec.append(np.array(data_in[i])-np.array(data_in[i-1]))
    
    tot_len = sum([np.linalg.norm(vec) for vec in all_segment_vec])
    step_size = tot_len/tot_steps
    temp_out = [data_in[0]]
    
    for vec in all_segment_vec:
        vec_len = np.linalg.norm(vec)
        steps = int(tot_steps*vec_len/tot_len)
        for k in range(steps):
            temp_out.append(temp_out[-1]+vec*step_size/vec_len)

    n_sub = int(float(tot_steps)/(n_points-1)) 
    data_out = temp_out[0::n_sub]
    if np.linalg.norm(data_out[-1]-data_in[-1]) < tolerance*tot_len/tot_steps:
        data_out.pop()
    data_out.append(data_in[-1])
    
    return np.array(data_out)
