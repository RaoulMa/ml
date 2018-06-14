/*
Author: Raoul Malm
Description: Counting inversions
 Inversion Count for an array indicates – how far (or close) the array is from being sorted.
 If array is already sorted then inversion count is 0. If array is sorted in reverse order
 that inversion count is the maximum.
 Formally speaking, two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j
 The sequence 2, 4, 1, 3, 5 has three inversions (2, 1), (4, 1), (4, 3).
*/

#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

long int  _mergeSort(vector<int> &arr, vector<int> &temp, int left, int right);
long int merge(vector<int> &arr, vector<int> &temp, int left, int mid, int right);

int mergeSort(vector<int> &arr, int array_size){
    /* This function sorts the input array and returns the
     number of inversions in the array */
    vector<int> temp(array_size);
    return _mergeSort(arr, temp, 0, array_size - 1);
}

long int _mergeSort(vector<int> &arr, vector<int> &temp, int left, int right){
    /* An auxiliary recursive function that sorts the input array and
     returns the number of inversions in the array. */

    int mid = 0;
    long int inv_count = 0;
    if (right > left)
    {
        /* Divide the array into two parts and call _mergeSortAndCountInv()
         for each of the parts */
        mid = (right + left)/2;
        
        /* Inversion count will be sum of inversions in left-part, right-part
         and number of inversions in merging */
        inv_count  = _mergeSort(arr, temp, left, mid);
        inv_count += _mergeSort(arr, temp, mid+1, right);
        
        /*Merge the two parts*/
        inv_count += merge(arr, temp, left, mid+1, right);
    }
    return inv_count;
}

long int merge(vector<int> &arr, vector<int> &temp, int left, int mid, int right){
    /* This funt merges two sorted arrays and returns inversion count in
     the arrays.*/

    int i, j, k;
    long int inv_count = 0;
    
    i = left; /* i is index for left subarray*/
    j = mid;  /* j is index for right subarray*/
    k = left; /* k is index for resultant merged subarray*/
    while ((i <= mid - 1) && (j <= right))
    {
        if (arr[i] <= arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
            
            /*this is tricky -- see above explanation/diagram for merge()*/
            inv_count = inv_count + (mid - i);
        }
    }
    
    /* Copy the remaining elements of left subarray
     (if there are any) to temp*/
    while (i <= mid - 1)
    temp[k++] = arr[i++];
    
    /* Copy the remaining elements of right subarray
     (if there are any) to temp*/
    while (j <= right)
    temp[k++] = arr[j++];
    
    /*Copy back the merged elements to original array*/
    for (i=left; i <= right; i++)
        arr[i] = temp[i];
    
    return inv_count;
}

long int getInvCount(vector<int>& arr, int array_size){
    /* Brute Force Method*/
    long int inv_count = 0;
    int n = array_size;

    for (int i = 0; i < n - 1; i++)
        for (int j = i+1; j < n; j++)
            if (arr[i] > arr[j])
                inv_count++;
    return inv_count;
}

void printArray(vector<int> arr, int array_size){
    /* Function to print an array */
    for (int i=0; i < array_size; i++){
        cout << arr[i] << " ";
    }
}

int main(int argv, char** args){
    
    cout << "\n--Count Number of Inversions --\n" << endl;
    int vv[] = {1, 20, 6, 4, 5, 8, 9, 2, 3, 28};
    vector<int> arr(&vv[0], &vv[10]);
    
    cout << "Test Array: ";
    printArray(arr, arr.size());
    cout << endl;
    cout << "Brute Force Method: ";
    cout << "Number of inversions = " << getInvCount(arr, arr.size()) <<endl;
    
    cout << "Recursive Method: ";
    cout << "Number of inversions = " << mergeSort(arr, arr.size()) <<endl;

    vector<int> numbers;
    int number;  //Variable to hold each number as it is read
    ifstream in("IntegerArray.txt",ios::in);

    while (in >> number)
        numbers.push_back(number); //Add the number to the end of the array
    in.close(); // close file stream
    
    cout << "\nRead in file IntegerArray.txt: Array of size " << numbers.size() << endl;

    //Display the numbers
    cout << "Show the first 5 numbers:\n";
    for (int i=0; i<5; i++) {
        cout << numbers[i] << ' ';
    }
    
    vector<int> numbers_copy = numbers;
    cout << "\nBrute Force Method: ";
    cout << "Number of inversions = " << getInvCount(numbers, numbers.size());
    
    cout << "\nApply the Recursive Method: ";
    cout << "Number of inversions = " << mergeSort(numbers, numbers.size()) << endl;
    
    
    //cin.get(); //Keep program open until "enter" is pressed
    
    return 0;
}



