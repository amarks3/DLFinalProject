from preprocess import *

def main(): 
    i_val = 11 #change this from 1-51 
    data, labels = get_data(i_val)
    print("num of slices: ", len(data))

if __name__ == '__main__':
	main()