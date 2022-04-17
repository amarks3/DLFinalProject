from preprocess import *

def main(): 
    i_val = 11 #change this from 1-51 
    image_slices = get_data(i_val)
    print("num of slices: ", len(image_slices))

if __name__ == '__main__':
	main()