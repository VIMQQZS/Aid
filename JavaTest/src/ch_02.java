// import java.util.Scanner;

// public class ch_02{
//     public static void main(String[] args){
//         Scanner sc = new Scanner(System.in);
//         int end = sc. nextInt();
//         int arr[] = new int[end];
//         for(int i = 0;i < end;i++){
// 			arr[i] = sc. nextInt();
//         }
//         int num = sc. nextInt();
//         int index = BSearch(0, end/2, end-1, arr, num);
//         insert(index, arr, num, end);
//     }
// 	public static void BSearch(int start,int mid,int end,int[] arr,int num){
//         if(start == end){
//             insert()
//             return;
//         }
//         if(arr[mid]<=num){
//             start = mid+1;
//             mid = (start+end) /2;
// 			BSearch(start,mid,end,arr,num);
//         }
//         else{
//             end = mid;
//             mid = (start+end) /2;
//             BSearch(start,mid,end,arr,num);
//         }
//     }
//     public static void insert(int index, int[] arr, int num, int end){
//         for(int i=end+1;i>index+1;i--){
//             arr[i+1]=arr[i];
//         }
//         arr[index]=num;
//         return;
//     }
// }
public class ch_02{
    public static void fun1() {
        int i = 2;//		
        switch(i){//			
            case 1://			
            System.out.print(1+"\t");//			
            case 2://			
            System.out.print(2+"\t");//			
            case 3://			
            System.out.print(3+"\t");//			
            default://			
            System.out.println("总之不是 1 2 3");//		
        }//	
    }//	
    public static void fun2() {//		
        int i = 2;//		
        switch(i) {//			
            default://			
            System.out.println("总之不是 1 2 3");//			
            case 1://			
            System.out.print(1+"\t");//			
            case 2://			
            System.out.print(2+"\t");//			
            case 3://			
            System.out.print(3+"\t");//		
        }//	
    }//	
    public static void fun3() {//		
        int i = 56;//		
        switch(i) {//			
            default://			
            System.out.println("总之不是 1 2 3");//			
            case 1://			
            System.out.print(1+"\t");//			
            case 2://			
            System.out.print(2+"\t");//			
            case 3://			
            System.out.print(3+"\t");//		
        }//	上面的四条语句都执行
    }//	
    public static void main(String[] args) {//		
        fun1();//		
        fun2();//		
        fun3();//	
    }

	// public static void main(String[] args) {
    //     int x=20,y=6;
    //     boolean result=(++x==33&&(y++>6));
    //     System.out.printf("x=%d,y=%d\n",x,y);
    //     result = (x++==33) & (y++>6);
    //     System.out.printf("x=%d,y=%d\n",x,y);
    // }
}