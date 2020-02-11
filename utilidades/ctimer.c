
////////////////////////////////////////////////////////////////////////////
//
// ctimer
//
// Rutina para la medición de tiempos
//
// Entrada: Ninguno
// Salida: elapsed, milisegundos de tiempo real transcurridos
//                  desde la primera vez que se llamó a la rutina
//         ucpu,    milisegundos consumidos realmente de cpu desde la 
//                  puesta en marcha del programa
//         scpu,    milisegundos consumidos realmente de cpu en tareas
//                  de sistema desde la puesta en marcha del programa.
// Retorno: 0, Ok, 
// 
// La rutina debe ejecutarse al menos dos veces. La primera pone en marcha
// el cronómetro, y devuelve 0 en elapsed. Los parámetros ucpu y scpu 
// devuelven el valor correcto. La segunda vez que se ejecute la función devuelve 
// en elapsed el tiempo real transcurrido entre la primera llamada y ésta.
// La tercera y siguientes llamadas devuelven el tiempo transcurrido entre
// la primera y ésta.
//
// Ejemplo:
//
//  double elapsed, ucpu, scpu;
//
//  ctimer(&elapsed, &ucpu, &scpu);
//  for (i=0;i<n;i++) {
//     for (j=0;j<n;j++) {
//       x[i] = A[i][j] * b[j];
//     }
//  }
//  ctimer(&elapsed, &ucpu, &scpu);
//  printf("El producto x=A*b ha necesitado %f milisegundos\n", elapsed);
//
//////////////////////////////////////////////////////////////////////////////


#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <unistd.h>

static double firstcall=0.0;

int ctimer(double *elapsed, double *ucpu, double *scpu ) {

  struct timeval tm;
  struct timezone tz;
  struct tms sistema;
  double usegs;

  gettimeofday(&tm, &tz);
  times(&sistema);

  usegs = tm.tv_usec+tm.tv_sec*1E6;

  if (firstcall)  {
    *elapsed = usegs - firstcall;
    firstcall = 0.0;
  } else {
    *elapsed = 0.0;
    //*ucpu = tm.tv_usec;
    //*scpu = ;
    firstcall = usegs;
  }

  *elapsed = *elapsed/1E6;
  *ucpu = (double)sistema.tms_utime/(double)CLOCKS_PER_SEC*1E4;
  *scpu = (double)sistema.tms_stime/(double)CLOCKS_PER_SEC*1E4;

  return 0;
} /* end ctimer */
