//gedit filename.cpp 
//gcc -fopenmp filename.cpp
//./a.out

#include<iostream>
#include<omp.h>
#include<bits/stdc++.h>

using namespace std;


    



void sequential_bubble_sort(int arr[],int size){
    int array[size];
    for(int i = 0 ; i < size; i++){
        array[i] = arr[i];
    }

    double start = omp_get_wtime();
    for(int i = 0; i < size - 1; i ++){
        for(int j = 0; j < size - i - 1; j++){
            if(array[j] > array[j+1]){
                swap(array[j],array[j+1]);
            }
        }
    }
    double end = omp_get_wtime();
    cout << "Sequential Bubble Sort:\n";
    // for(int i = 0 ; i < size; i++){
    //     cout << array[i] << " ";
    // }
    cout << endl;
    cout << "Time Required: " << end - start << endl;

}

void parallel_bubble_sort(int arr[],int size){
    int array[size];
    for(int i = 0 ; i < size; i++){
        array[i] = arr[i];
    }
    double start = omp_get_wtime();
    for(int k = 0; k < size;k ++){
        if(k % 2 == 0){
            #pragma omp parallel for
                for(int i = 1; i < size - 1; i += 2){
                    if(array[i] > array[i+1]){
                        swap(array[i],array[i+1]);
                    }
                }
        }
        else{
            #pragma omp parallel for
                for(int i = 0; i < size - 1; i += 2){
                    if(array[i] > array[i+1]){
                        swap(array[i],array[i+1]);
                    }
                }
        }
    }
    double end = omp_get_wtime();
    cout << "Parallel Bubble Sort:\n";
    // for(int i = 0 ; i < size; i++){
    //     cout << array[i] << " ";
    // }
    cout << endl;
    cout << "Time Required: " << end - start << endl;
}

void merge(int array[],int low, int mid, int high,int size){
    int temp[size];
    int i = low;
    int j = mid + 1;
    int k = 0;
    while((i <= mid) && (j <= high)){
        if(array[i] >= array[j]){
            temp[k] = array[j];
            k++;
            j++;
        }
        else{
            temp[k] = array[i];
            k++;
            i++;
        }
    }
    while(i <= mid){
        temp[k] = array[i];
        k++;
        i++;
    }
    while(j <= high){
        temp[k] = array[j];
        k++;
        j++;
    }

    k = 0;
    for(int i = low;i <= high;i++){
        array[i] = temp[k];
        k++;
    }
}

void mergesort(int array[],int low,int high,int size){
    if(low < high){
        int mid = (low + high) / 2;
        mergesort(array,low,mid,size);
        mergesort(array,mid+1,high,size);
        merge(array,low,mid,high,size);
    }
}

void perform_merge_sort(int arr[],int size){
    int array[size];
    for(int i = 0 ; i < size; i++){
        array[i] = arr[i];
    }
    double start = omp_get_wtime();
    mergesort(array,0,size-1,size);
    double end = omp_get_wtime();
    cout << "Merge Sort:\n";
    // for(int i = 0 ; i < size; i++){
    //     cout << array[i] << " ";
    // }
    cout << endl;
    cout << "Time Required: " << end - start << endl;
}

void p_mergesort(int array[],int low,int high,int size){
    if(low < high){
        int mid = (low + high) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
                p_mergesort(array,low,mid,size);
            #pragma omp section
                p_mergesort(array,mid+1,high,size);
        }
        merge(array,low,mid,high,size);
    }
}

void perform_p_merge_sort(int arr[],int size){
    int array[size];
    for(int i = 0 ; i < size; i++){
        array[i] = arr[i];
    }
    double start = omp_get_wtime();
    p_mergesort(array,0,size-1,size);
    double end = omp_get_wtime();
    cout << "Parallel Merge Sort:\n";
    // for(int i = 0 ; i < size; i++){
    //     cout << array[i] << " ";
    // }
    cout << endl;
    cout << "Time Required: " << end - start << endl;
}



int main(int argc, char const *argv[])
{
    int SIZE;
    int MAX = 1000;
    cout << "Enter size of array: ";
    cin >> SIZE;
    int array[SIZE];
    for(int i = 0 ; i < SIZE; i ++){
        array[i] = rand() % MAX;
    }
    // cout << "Initial Array:\n";
    // for(int i = 0 ; i < SIZE; i++){
    //     cout << array[i] << " ";
    // }
    cout << endl;
    sequential_bubble_sort(array,SIZE);
    parallel_bubble_sort(array,SIZE);
    perform_merge_sort(array,SIZE);
    perform_p_merge_sort(array,SIZE);
    return 0;
}

//_____________________________________________________________________________________
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
using namespace std;

// Parallel Bubble Sort function
void parallelBubbleSort(int *array, int n)
{
    int i, j;

#pragma omp parallel for private(i, j) shared(array)
    for (i = 0; i < n - 1; i++)
    {
        for (j = 0; j < n - i - 1; j++)
        {
            if (array[j] > array[j + 1])
            {
                // Swap elements
                int temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

// Parallel Merge Sort function

void merge(int *array, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temp arrays
    int *L = new int[n1];
    int *R = new int[n2];

    // Copy data to temp arrays L[] and R[]

    for (i = 0; i < n1; i++)
        L[i] = array[l + i];

    for (j = 0; j < n2; j++)
        R[j] = array[m + 1 + j]

     // Merge the temp arrays back into array [l..r]

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            array[k] = L[i];
            i++;
        }
        else
        {
            array[k] = R[j];
            j++;
        }
        k++;
    }
    // Copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        array[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        array[k] = R[j];
        j++;
        k++;
    }
    delete[] L;
    delete[] R;
}

void parallelMergeSort(int *array, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections {
        #pragma omp section
        parallelMergeSort(array, l, m);

        #pragma omp section
        parallelMergeSort(array, m + 1, r);
    }
    merge(array, l, m, r);
}
}

int main()
{
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;
    int *array = new int[n];
    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        array[i] = rand() % 100;
    }

    cout << "Original Array: ";
    for (int i = 0; i < n; i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
    int choice;

    cout << "Enter 1 for Parallel Bubble Sort or 2 for Parallel Merge Sort:";
    cin >> choice;
    if (choice == 1)
    {
        parallelBubbleSort(array, n);
    }
    else if (choice == 2)
    {
        parallelMergeSort(array, 0, n - 1);
    }
    else
    {
        cout << "Invalid choice. Exiting program." << endl;
        return 0;
    }
    cout << "Sorted Array: ";
    for (int i = 0; i < n; i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
    delete[] array;
    return 0;
}
