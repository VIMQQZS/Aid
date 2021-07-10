import java.util.Scanner;

public class oj{
    // 
    // public static void main(String[] args){
		
    // }

	// 

    // 矩阵标兵
	// public static void main(String[] args){
    //     Scanner sc = new Scanner(System.in);
    //     int n = sc.nextInt();
    //     int[][] arr = new int[n][6];
    //     //int grade[] = new int[n];
    //     int sum[] = new int[n];
    //     for(int i=0;i<n;i++){
    //         sum[i]=0;
	// 		for(int j=0;j<6;j++){
    //             arr[i][j]=sc.nextInt() ;
    //         	if(j!=0)sum[i] = sum[i]+arr[i][j];
    //         } 
    //     }
    //     int max = sum[0];
    //     int index = arr[0][0];
    //     for(int i=1;i<n;i++){
    //         if(sum[i]>max){
    //             max=sum[i];
    //             index=arr[i][0];
    //         }
    //     }
    //     System. out. println("No:"+index);
    //     System. out. println();
    // }

    // 矩阵转置
    // public static void main(String[] args){
    //     Scanner sc = new Scanner(System.in);
    //     int n = sc.nextInt();
    //     int m = sc. nextInt();
    //     int arr1[][] = new int[n][m];
    //     int arr2[][] = new int[m][n];
    //     for(int i=0;i<n;i++){
    //         for(int j=0;j<m;j++){
    //             arr1[i][j] = sc. nextInt();
    //         }
    //     }
    //     for(int i=0;i<m;i++){
    //         for(int j=0;j<n;j++){
	// 			arr2[i][j]=arr1[j][i];
    //         	System. out. printf("%6d",arr2[i][j]);
    //         }
    //         System. out. println() ;
    //     }
    //     System.out.println();
    // }


//     public static void main(String[] args){
//         Scanner sc = new Scanner(System.in);
//         while(true){
//             int num = sc.nextInt();
//             boolean flag = false;
//             int inv = inv(num);
//             if(num==inv){
//                 System.out.println(num);
//                 continue;
//             }
//         	for(int i=0;i<10;i++){
//                 num = num + inv;
//                 inv = inv(num);
//                 // System. out. println(inv);
//                 if(num==inv){
//                     System.out.println(num);
//                     flag = true;
//                     break;
//                 }
//             }
//             if(flag==false){
//                 System. out. println("not");
//             	continue;
//             }
//         }
//     }
//     public static int inv(int num){
//         int inv = 0;
//         while(num!=0){
//             inv = inv *10+ num %10;
//             num = num/10;
//         }
//         return inv;
//     }
}


