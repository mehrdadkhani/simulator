
#include <chrono>
#include <cmath>
#include "pie_packet_queue.hh"
#include "timestamp.hh"
#include <unistd.h>
#include <stdio.h>
using namespace std;

#define DQ_COUNT_INVALID   (uint32_t)-1


//Some Hack Here...
#define DROPRATE_UPDATE_PERIOD 40
#define NN_UPDATE_PERIOD 100
#define DROPRATE_BOUND 0.3
#define SWEEP_TIME 0




double * _drop_prob = NULL;
double rl_drop_prob = 0.0;
unsigned int _size_bytes_queue = 0;
uint32_t * _current_qdelay = NULL;





#define MAXLAYER 5
struct NeuralNetwork{
	int dim_layer[MAXLAYER];
	float **w;
	float **b;	
};

#define STATE_DIM  24
struct State{
    float s[STATE_DIM];
};



#define STATE_RING_SIZE 16
struct StateRing{
    struct State ring[STATE_RING_SIZE];
    int counter;
};










struct NeuralNetwork NN_A,NN_B;
struct NeuralNetwork * NN_cur = NULL; 
pthread_mutex_t swap_lock;


struct StateRing stateRing;
struct State * state_cur = NULL;


float ring_avg(float* data, int start, int end)
{
    float sum = 0;
    int ptr = start;
    while(ptr <= end)
    {
        sum += data[ptr % 256];
        ptr ++;
    }
    return sum / (end - start + 1);
}


void UpdateState(float qdelay, float dprate)
{
    static float qdelay_list[256];
    static float dprate_list[256];
    static int ptr = 256;

    state_cur = &(stateRing.ring[stateRing.counter % STATE_RING_SIZE]);
    stateRing.counter ++;
    
    qdelay_list[ptr % 256] = qdelay;
    dprate_list[ptr % 256] = dprate;
    
    state_cur->s[0] = ring_avg(qdelay_list,ptr-63, ptr);
    state_cur->s[1] = ring_avg(dprate_list,ptr-63, ptr);
    state_cur->s[2] = ring_avg(qdelay_list,ptr-31, ptr);
    state_cur->s[3] = ring_avg(dprate_list,ptr-31, ptr);
    state_cur->s[4] = ring_avg(qdelay_list,ptr-15, ptr);
    state_cur->s[5] = ring_avg(dprate_list,ptr-15, ptr);
    state_cur->s[6] = ring_avg(qdelay_list,ptr-7, ptr);
    state_cur->s[7] = ring_avg(dprate_list,ptr-7, ptr);

    for(int i = 0; i<8; i++)
    {
        state_cur->s[i*2+8] = qdelay_list[(ptr - 7 + i)%256];
        state_cur->s[i*2+8+1] = dprate_list[(ptr - 7 + i)%256];
    }
    
    ptr ++;    
}


void printStateRing()
{
    FILE *fp = NULL;

    fp = fopen("/home/rl/Project/rl-qm/mahimahiInterface/statering.txt","w");

    if(fp == NULL)
    {
        printf("Failed to open statering file\n");
    }
    else
    {
        for(int i = 0; i<STATE_RING_SIZE - 1; i++)
        {
            fprintf(fp,"%d", stateRing.counter - STATE_RING_SIZE + 1+ i);
            for(int j = 0; j< STATE_DIM; j++)
            {
               fprintf(fp," %f",stateRing.ring[(stateRing.counter - STATE_RING_SIZE + 1+ i)%STATE_RING_SIZE].s[j]);
			   
            }
		
            fprintf(fp," %f",stateRing.ring[(stateRing.counter - STATE_RING_SIZE + 1+ i + 1)%STATE_RING_SIZE].s[22]);
            fprintf(fp," %f",stateRing.ring[(stateRing.counter - STATE_RING_SIZE + 1+ i + 1)%STATE_RING_SIZE].s[23]);

            fprintf(fp,"\n");
        }         


        fclose(fp);
    }
    



}





void initNN()
{
	pthread_mutex_init(&swap_lock, NULL);
	//TODO Load Model
	NN_A.dim_layer[0] = 24;
	NN_A.dim_layer[1] = 150;
	NN_A.dim_layer[2] = 200;
	NN_A.dim_layer[3] = 150;
	NN_A.dim_layer[4] = 1;

	NN_B.dim_layer[0] = 24;
	NN_B.dim_layer[1] = 150;
	NN_B.dim_layer[2] = 200;
	NN_B.dim_layer[3] = 150;
	NN_B.dim_layer[4] = 1;

	NN_A.w = (float**)malloc(sizeof(float*) * MAXLAYER);
	NN_B.w = (float**)malloc(sizeof(float*) * MAXLAYER);
	NN_A.b = (float**)malloc(sizeof(float*) * MAXLAYER);
	NN_B.b = (float**)malloc(sizeof(float*) * MAXLAYER);

	for(int i = 0; i< MAXLAYER-1; i++) NN_A.w[i] = (float*)malloc(sizeof(float) * NN_A.dim_layer[i] * NN_A.dim_layer[i+1]);
 	for(int i = 0; i< MAXLAYER-1; i++) NN_B.w[i] = (float*)malloc(sizeof(float) * NN_B.dim_layer[i] * NN_B.dim_layer[i+1]);
 	for(int i = 0; i< MAXLAYER-1; i++) NN_A.b[i] = (float*)malloc(sizeof(float) * NN_A.dim_layer[i+1]);
	for(int i = 0; i< MAXLAYER-1; i++) NN_B.b[i] = (float*)malloc(sizeof(float) * NN_B.dim_layer[i+1]);
 	


	NN_cur = &NN_A;
    stateRing.counter = STATE_RING_SIZE;
}


void RunNN(float *input, float* output)
{
	float bufA[256];
	float bufB[256];

	float * pin;
	float * pout;

	pin = input;
	pout = bufA;


	if(NN_cur == NULL) initNN();

	pthread_mutex_lock(&swap_lock);
	
		for(int i = 0; i< MAXLAYER-1; i++)
		{
			
			for(int k=0; k<NN_cur->dim_layer[i+1]; k++)
			{
				pout[k] = NN_cur->b[i][k];
			}

			for(int j=0; j< NN_cur->dim_layer[i]; j++)
			{
				for(int k=0; k<NN_cur->dim_layer[i+1]; k++)
				{
					pout[k] = pout[k] + pin[j] * NN_cur->w[i][j*NN_cur->dim_layer[i+1] + k];
					//if(i==0 && j==0 && k<10) printf("%f ", NN_cur->w[i][j*NN_cur->dim_layer[i+1] + k]);
				}
			}

			if(i != MAXLAYER-2)
				for(int k=0; k<NN_cur->dim_layer[i+1]; k++)
				{
					//printf("%d %f\n",i,pout[k]);
					pout[k] = pout[k] > 0? pout[k]: 0;
					
				}

			
			pin = pout;
			if(pout == bufA) pout = bufB;
			else pout = bufA;

			if(i == MAXLAYER-3) pout = output;
		}

	pthread_mutex_unlock(&swap_lock);

	pin[0] = 1/(1.0+exp(-pin[0])) * DROPRATE_BOUND;


}


void SwapNN()
{
	pthread_mutex_lock(&swap_lock);

		if(NN_cur == &NN_A) NN_cur = &NN_B;
		else NN_cur = &NN_A;

	pthread_mutex_unlock(&swap_lock);
}



//Update Neure Network Model (Load from file)
void* NN_thread(void* context)
{
	int ret = 0;
	struct NeuralNetwork * NN;
	int total_read = 0;
	while(true)
	{
		usleep(100*1000); // Every 100ms
		//printf("This is NN thread Load \n");

		//TODO Load Parameter
		if(NN_cur == &NN_A)
		{
			NN = &NN_B;
		}
		else
		{
			NN = &NN_A;
		}
		ret = 0;
		FILE *fp = NULL;
		fp = fopen("/home/rl/Project/rl-qm/mahimahiInterface/NN.txt","r");
		total_read = 0;
		if(fp == NULL)
		{
			printf("Failed to load parameters\n");
			usleep(1000*NN_UPDATE_PERIOD);//
		}
		else
		{
			total_read = 0;
			//printf("Load Parameters\n");
			for(int i=0;i<MAXLAYER-1;i++)
			{
				for(int j=0; j< NN->dim_layer[i] * NN->dim_layer[i+1]; j++)
				{
					ret += fscanf(fp,"%f",&(NN->w[i][j]));
					total_read ++;
				}
				for(int j=0; j< NN->dim_layer[i+1]; j++)
				{
					ret += fscanf(fp,"%f",&(NN->b[i][j]));
					total_read ++;
				}

			}
			
			fclose(fp);
		}
		
		//printf("Swap NN Parameters\n");
		//Swap Buffer
		if(ret == total_read)
			SwapNN();
		else
			printf("Load NN Parameters failed %d/%d\n",ret, total_read);

	}
	return context;
}


//Update Drop Rate 
void* UpdateDropRate_thread(void* context)
{
	//float state[24];
	float action;
    static float action_old = 0;

	static int sweep = 1;
	static int step = 0;
	static float sweep_dp = 0.00001;
	
	while(true)
	{
		step++;
		if(step * DROPRATE_UPDATE_PERIOD > SWEEP_TIME * 1000 ) sweep = 0;

		//uint64_t now = timestamp();
		usleep(1000*DROPRATE_UPDATE_PERIOD);
	
		UpdateState((float)(*_current_qdelay), action_old);
	
		state_cur = &(stateRing.ring[stateRing.counter % STATE_RING_SIZE]);
	
		RunNN(state_cur->s, &action);

		//for(int i = 0; i<24; i++) state[i] = i;
		//state_cur = &(stateRing.ring[stateRing.counter % STATE_RING_SIZE]);
		//RunNN(state_cur->s, &action);
        
        //UpdateState((float)(*_current_qdelay), action_old);


        action_old = action;

		if(sweep == 0)
		{
			rl_drop_prob = action * 0.75 + rl_drop_prob * 0.25;
		}
		else
		{
			if(step % 50 == 0)
			{
				if(sweep_dp < 0.01)
				{
					sweep_dp *= 2;
				}
				else
					sweep_dp += 0.005;
				
				if(sweep_dp > 0.15)
					sweep_dp += 0.005;


				if(sweep_dp > DROPRATE_BOUND)
				{
					sweep_dp = 0.00001;
				}

				//rl_drop_prob = sweep_dp;
				//action_old = sweep_dp;		
			}

			rl_drop_prob = sweep_dp;
			action_old = sweep_dp;		
		}

		//printf("Current Drop Rate %.6lf Queue Size %u Bytes, Qdelay %u  Action %f\n", *_drop_prob, _size_bytes_queue,*_current_qdelay, action);
        printStateRing();
		//uint64_t interval = timestamp() - now;

		//printf("%lu\n",interval);
	}
return context;
}




PIEPacketQueue::PIEPacketQueue( const string & args )
  : DroppingPacketQueue(args),
    qdelay_ref_ ( get_arg( args, "qdelay_ref" ) ),
    max_burst_ ( get_arg( args, "max_burst" ) ),
    alpha_ ( 0.125 ),
    beta_ ( 1.25 ),
    t_update_ ( 10 ),
    dq_threshold_ ( 16384 ),
    drop_prob_ ( 0.0 ),
    burst_allowance_ ( 0 ),
    qdelay_old_ ( 0 ),
    current_qdelay_ ( 0 ),
    dq_count_ ( DQ_COUNT_INVALID ),
    dq_tstamp_ ( 0 ),
    avg_dq_rate_ ( 0 ),
    uniform_generator_ ( 0.0, 1.0 ),
    prng_( random_device()() ),
    last_update_( timestamp() ),
		NN_t( 0 ),
		DP_t( 0 )
{
  if ( qdelay_ref_ == 0 || max_burst_ == 0 ) {
    throw runtime_error( "PIE AQM queue must have qdelay_ref and max_burst parameters" );
  }

	
}

void PIEPacketQueue::enqueue( QueuedPacket && p )
{
	static int counter = 0;

	counter++;
	if(counter == 10)
	{
    initNN();
		pthread_create(&(this->NN_t),NULL,&NN_thread,NULL);
		pthread_create(&(this->DP_t),NULL,&UpdateDropRate_thread,NULL);
		printf("Create Pthread!\n");
	}
  calculate_drop_prob();
	
  _drop_prob = &(this->drop_prob_);
	_current_qdelay = &(this->current_qdelay_);
	_size_bytes_queue = size_bytes();

  //printf("%u\n",size_bytes());

  if ( ! good_with( size_bytes() + p.contents.size(),
		    size_packets() + 1 ) ) {
    // Internal queue is full. Packet has to be dropped.
    return;
  } 

  if (!drop_early() ) {
    //This is the negation of the pseudo code in the IETF draft.
    //It is used to enqueue rather than drop the packet
    //All other packets are dropped
    accept( std::move( p ) );
  }

  assert( good() );
}

//returns true if packet should be dropped.
bool PIEPacketQueue::drop_early ()
{
  /*
  if ( burst_allowance_ > 0 ) {
    return false;
  }

  if ( qdelay_old_ < qdelay_ref_/2 && drop_prob_ < 0.2) {
    return false;        
  }

  if ( size_bytes() < (2 * PACKET_SIZE) ) {
    return false;
  }
  */

  double random = uniform_generator_(prng_);

  //if ( random < drop_prob_ ) {
  if ( random < rl_drop_prob ) {
    return true;
  }
  else
    return false;
}

QueuedPacket PIEPacketQueue::dequeue( void )
{
  QueuedPacket ret = std::move( DroppingPacketQueue::dequeue () );
  uint32_t now = timestamp();

  if ( size_bytes() >= dq_threshold_ && dq_count_ == DQ_COUNT_INVALID ) {
    dq_tstamp_ = now;
    dq_count_ = 0;
  }

  if ( dq_count_ != DQ_COUNT_INVALID ) {
    dq_count_ += ret.contents.size();

    if ( dq_count_ > dq_threshold_ ) {
      uint32_t dtime = now - dq_tstamp_;

      if ( dtime > 0 ) {
	uint32_t rate_sample = dq_count_ / dtime;
	if ( avg_dq_rate_ == 0 ) 
	  avg_dq_rate_ = rate_sample;
	else
	  avg_dq_rate_ = ( avg_dq_rate_ - (avg_dq_rate_ >> 3 )) +
		     (rate_sample >> 3);
                
	if ( size_bytes() < dq_threshold_ ) {
	  dq_count_ = DQ_COUNT_INVALID;
	}
	else {
	  dq_count_ = 0;
	  dq_tstamp_ = now;
	} 

	if ( burst_allowance_ > 0 ) {
	  if ( burst_allowance_ > dtime )
	    burst_allowance_ -= dtime;
	  else
	    burst_allowance_ = 0;
	}
      }
    }
  }

  calculate_drop_prob();

  return ret;
}

void PIEPacketQueue::calculate_drop_prob( void )
{
  uint64_t now = timestamp();
	
  //We can't have a fork inside the mahimahi shell so we simulate
  //the periodic drop probability calculation here by repeating it for the
  //number of periods missed since the last update. 
  //In the interval [last_update_, now] no change occured in queue occupancy 
  //so when this value is used (at enqueue) it will be identical
  //to that of a timer-based drop probability calculation.
  while (now - last_update_ > t_update_) {
    bool update_prob = true;
    qdelay_old_ = current_qdelay_;

    if ( avg_dq_rate_ > 0 ) 
      current_qdelay_ = size_bytes() / avg_dq_rate_;
    else
      current_qdelay_ = 0;

    if ( current_qdelay_ == 0 && size_bytes() != 0 ) {
      update_prob = false;
    }

    double p = (alpha_ * (int)(current_qdelay_ - qdelay_ref_) ) +
      ( beta_ * (int)(current_qdelay_ - qdelay_old_) );

    if ( drop_prob_ < 0.01 ) {
      p /= 128;
    } else if ( drop_prob_ < 0.1 ) {
      p /= 32;
    } else  {
      p /= 16;
    } 

    drop_prob_ += p;

    if ( drop_prob_ < 0 ) {
      drop_prob_ = 0;
    }
    else if ( drop_prob_ > 1 ) {
      drop_prob_ = 1;
      update_prob = false;
    }

        
    if ( current_qdelay_ == 0 && qdelay_old_==0 && update_prob) {
      drop_prob_ *= 0.98;
    }
        
    burst_allowance_ = max( 0, (int) burst_allowance_ -  (int)t_update_ );
    last_update_ += t_update_;

    if ( ( drop_prob_ == 0 )
	 && ( current_qdelay_ < qdelay_ref_/2 ) 
	 && ( qdelay_old_ < qdelay_ref_/2 ) 
	 && ( avg_dq_rate_ > 0 ) ) {
      dq_count_ = DQ_COUNT_INVALID;
      avg_dq_rate_ = 0;
      burst_allowance_ = max_burst_;
    }

  }
}
