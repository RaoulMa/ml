/*
Like Merge Sort, QuickSort is a Divide and Conquer algorithm. It picks an element as pivot and partitions the given array around the picked pivot. There are many different versions of quickSort that pick pivot in different ways.

- Always pick first element as pivot.
- Always pick last element as pivot (implemented below)
- Pick a random element as pivot.
- Pick median as pivot.

The key process in quickSort is partition(). Target of partitions is, given an array and an element x of array as pivot, put x at its correct position in sorted array and put all smaller elements (smaller than x) before x, and put all greater elements (greater than x) after x. All this should be done in linear time.
*/

#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

int num_comparisons = 0; // total number of comparisons
int pivot_type = 0;

void printArray(int arr[], int size){
    /* Function to print an array */
    int i;
    for (i=0; i < size; i++)
    cout << arr[i] << " ";
    
}

void swap(int* a, int* b){
    // A utility function to swap two elements
    int t = *a;
    *a = *b;
    *b = t;
}

int partition (int arr[], int left, int right){
    /* This function takes last element as pivot, places
     the pivot element at its correct position in sorted
     array, and places all smaller (smaller than pivot)
     to left of pivot and all greater elements to right
     of pivot */
    
    //cout << endl;
    //cout << left << right << " ";
    //printArray(arr, right-left+1);
    //cout << endl;
    
    if (pivot_type==0) {
        
        int pivot = arr[left];    // left pivot
        int i = left + 1;
        
        for (int j = left+1; j <= right; j++){
            num_comparisons += 1;
            // If current element is smaller than or equal to pivot
            if (arr[j] < pivot){
                swap(&arr[i], &arr[j]);
                i++;
            }
        }
        swap(&arr[i-1], &arr[left]);
        //cout << left << right << " ";
        //printArray(arr, right-left+1);
        //cout << endl;
        return (i-1);
    }
    else if (pivot_type==1) {
        
        int pivot = arr[right];    // right pivot
        int i = (left - 1);  // Index of smaller element
        
        for (int j = left; j < right; j++){
            num_comparisons += 1;
            // If current element is smaller than or
            // equal to pivot
            if (arr[j] < pivot)
            {
                i++;    // increment index of smaller element
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[right]);
        return (i + 1);
    }
    return -1;
}

void quickSort(int arr[], int left, int right){
    /* The main function that implements QuickSort
     arr[] --> Array to be sorted,
     left  --> Starting index,
     right  --> Ending index */
    if (left < right){
        /* pi is partitioning index, arr[p] is now
         at right place */
        
        int pi = partition(arr, left, right);
        quickSort(arr, left, pi-1);
        quickSort(arr, pi+1, right);
    }
}

int main(){
    
    int arr[] = {3, 8, 2, 5, 1, 4, 7};
    int n = sizeof(arr)/sizeof(arr[0]);

    cout << "unsorted test array: ";
    printArray(arr, n);
    cout << endl;
    quickSort(arr, 0, n-1);
    cout << "sorted test array: ";
    printArray(arr, n);
    cout << endl;
    
    vector<int> v_arr;
    int number;  //Variable to hold each number as it is read
    ifstream in("QuickSort.txt",ios::in);
    
    while (in >> number)
    v_arr.push_back(number); //Add the number to the end of the array
    in.close(); // close file stream
    vector<int> v_arr_copy = v_arr;
    
    cout << "\nRead in file QuickSort.txt: Array of size " << v_arr.size() << endl;
    cout << "Show the first unsorted 10 numbers: ";
    printArray(&v_arr[0], 10);
    cout << endl;
    
    cout << endl << "using left pivot: ";
    cout << "show the first sorted 100 numbers: ";
    pivot_type = 0;
    num_comparisons = 0;
    quickSort(&v_arr[0], 0, v_arr.size());
    printArray(&v_arr[0], 100);
    cout << endl << "total number of comparisons: " << num_comparisons << endl;

    cout << endl << "using right pivot: ";
    cout << "show the first sorted 100 numbers: ";
    pivot_type = 1;
    num_comparisons = 0;
    quickSort(&v_arr_copy[0], 0, v_arr_copy.size());
    printArray(&v_arr_copy[0], 100);
    cout << endl << "total number of comparisons: " << num_comparisons << endl;

    

    
    return 0;
}
