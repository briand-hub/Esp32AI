/** Copyright (C) 2023 briand (https://github.com/briand-hub)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifndef BRIAND_AI_DEBUG
    #define BRIAND_AI_DEBUG 1 // DEBUG MODE (print to stdout calculus and other info)
#endif

#ifndef BRIAND_INCLUDE_H
#define BRIAND_INCLUDE_H

    /* All headers needed in library, includes the porting to other platforms */

    // C++ STL

    #include <iostream>
    #include <cstdio>
    #include <cmath>
    #include <memory>
    #include <vector>
    #include <map>
    #include <cstdlib>
    #include <cstring>
    #include <thread>
    #include <chrono>
    #include <algorithm>
    #include <unistd.h>
    #include <signal.h>
	#include <limits>
	#include <cassert>

    /* 
        Small code redefining in linux/windows platform used ESP functions and types in order to compile and test on other platforms
    */

    #if defined(ESP_PLATFORM)
        // Set BRIAND_PLATFORM for printing out current platform if needed
        #define BRIAND_PLATFORM "ESP32"

        // Here normal ESP Headers must be used

        #include "esp_log.h"
		#include "esp_random.h"
		#include "esp_timer.h"

    #elif defined(__linux__) | defined(_WIN32)
        // Set BRIAND_PLATFORM for printing out current platform if needed
        #if defined(__linux__)
            #define BRIAND_PLATFORM "Linux"
        #elif defined(_WIN32)
            #define BRIAND_PLATFORM "Windows"
        #endif

        // Define the entry point (app_main will never be called!)
        extern "C" void app_main();
    
        //
        // Here ESP types, objects, functions must be re-defined when needed.
        //


		using namespace std;

		// GPIOS and system basics

		typedef enum {
			GPIO_NUM_NC = -1,    /*!< Use to signal not connected to S/W */
			GPIO_NUM_0 = 0,     /*!< GPIO0, input and output */
			GPIO_NUM_1 = 1,     /*!< GPIO1, input and output */
			GPIO_NUM_2 = 2,     /*!< GPIO2, input and output */
			GPIO_NUM_3 = 3,     /*!< GPIO3, input and output */
			GPIO_NUM_4 = 4,     /*!< GPIO4, input and output */
			GPIO_NUM_5 = 5,     /*!< GPIO5, input and output */
			GPIO_NUM_6 = 6,     /*!< GPIO6, input and output */
			GPIO_NUM_7 = 7,     /*!< GPIO7, input and output */
			GPIO_NUM_8 = 8,     /*!< GPIO8, input and output */
			GPIO_NUM_9 = 9,     /*!< GPIO9, input and output */
			GPIO_NUM_10 = 10,   /*!< GPIO10, input and output */
			GPIO_NUM_11 = 11,   /*!< GPIO11, input and output */
			GPIO_NUM_12 = 12,   /*!< GPIO12, input and output */
			GPIO_NUM_13 = 13,   /*!< GPIO13, input and output */
			GPIO_NUM_14 = 14,   /*!< GPIO14, input and output */
			GPIO_NUM_15 = 15,   /*!< GPIO15, input and output */
			GPIO_NUM_16 = 16,   /*!< GPIO16, input and output */
			GPIO_NUM_17 = 17,   /*!< GPIO17, input and output */
			GPIO_NUM_18 = 18,   /*!< GPIO18, input and output */
			GPIO_NUM_19 = 19,   /*!< GPIO19, input and output */
			GPIO_NUM_20 = 20,   /*!< GPIO20, input and output */
			GPIO_NUM_21 = 21,   /*!< GPIO21, input and output */
			GPIO_NUM_26 = 26,   /*!< GPIO26, input and output */
			GPIO_NUM_27 = 27,   /*!< GPIO27, input and output */
			GPIO_NUM_28 = 28,   /*!< GPIO28, input and output */
			GPIO_NUM_29 = 29,   /*!< GPIO29, input and output */
			GPIO_NUM_30 = 30,   /*!< GPIO30, input and output */
			GPIO_NUM_31 = 31,   /*!< GPIO31, input and output */
			GPIO_NUM_32 = 32,   /*!< GPIO32, input and output */
			GPIO_NUM_33 = 33,   /*!< GPIO33, input and output */
			GPIO_NUM_34 = 34,   /*!< GPIO34, input and output */
			GPIO_NUM_35 = 35,   /*!< GPIO35, input and output */
			GPIO_NUM_36 = 36,   /*!< GPIO36, input and output */
			GPIO_NUM_37 = 37,   /*!< GPIO37, input and output */
			GPIO_NUM_38 = 38,   /*!< GPIO38, input and output */
			GPIO_NUM_39 = 39,   /*!< GPIO39, input and output */
			GPIO_NUM_40 = 40,   /*!< GPIO40, input and output */
			GPIO_NUM_41 = 41,   /*!< GPIO41, input and output */
			GPIO_NUM_42 = 42,   /*!< GPIO42, input and output */
			GPIO_NUM_43 = 43,   /*!< GPIO43, input and output */
			GPIO_NUM_44 = 44,   /*!< GPIO44, input and output */
			GPIO_NUM_45 = 45,   /*!< GPIO45, input and output */
			GPIO_NUM_46 = 46,   /*!< GPIO46, input mode only */
			GPIO_NUM_MAX,
		/** @endcond */
		} gpio_num_t;
		
		#define GPIO_MODE_DEF_DISABLE         (0b00000000)
		#define GPIO_MODE_DEF_INPUT           (0b00000001)    ///< bit mask for input
		#define GPIO_MODE_DEF_OUTPUT          (0b00000010)    ///< bit mask for output
		#define GPIO_MODE_DEF_OD              (0b00000100)    ///< bit mask for OD mode

		typedef enum {
			GPIO_MODE_DISABLE = GPIO_MODE_DEF_DISABLE,                                                         /*!< GPIO mode : disable input and output             */
			GPIO_MODE_INPUT = GPIO_MODE_DEF_INPUT,                                                             /*!< GPIO mode : input only                           */
			GPIO_MODE_OUTPUT = GPIO_MODE_DEF_OUTPUT,                                                           /*!< GPIO mode : output only mode                     */
			GPIO_MODE_OUTPUT_OD = ((GPIO_MODE_DEF_OUTPUT) | (GPIO_MODE_DEF_OD)),                               /*!< GPIO mode : output only with open-drain mode     */
			GPIO_MODE_INPUT_OUTPUT_OD = ((GPIO_MODE_DEF_INPUT) | (GPIO_MODE_DEF_OUTPUT) | (GPIO_MODE_DEF_OD)), /*!< GPIO mode : output and input with open-drain mode*/
			GPIO_MODE_INPUT_OUTPUT = ((GPIO_MODE_DEF_INPUT) | (GPIO_MODE_DEF_OUTPUT)),                         /*!< GPIO mode : output and input mode                */
		} gpio_mode_t;

		#define esp_restart() { printf("\n\n***** esp_restart() SYSTEM REBOOT. Hit Ctrl-C to restart main program.*****\n\n"); }
        #define gpio_set_level(n, l) { printf("\n\n***** GPIO %d => Level = %d*****\n\n", n, l); }
		#define gpio_get_level(n) GPIO_MODE_DISABLE
        #define gpio_set_direction(n, d) { printf("\n\n***** GPIO %d => Direction = %d*****\n\n", n, d); }

        
		// ERROR AND LOG FUNCTION

		#define ESP_OK 0
		#define ESP_FAIL -1
		#define ESP_ERR_NOT_FOUND -2
		#define ESP_ERR_NVS_NO_FREE_PAGES -3
		#define ESP_ERR_NVS_NEW_VERSION_FOUND -4

		typedef int esp_err_t;

		const char *esp_err_to_name(esp_err_t code);

		// LOGGING FUNCTIONS

		typedef enum esp_log_level {
			ESP_LOG_NONE,       /*!< No log output */
			ESP_LOG_ERROR,      /*!< Critical errors, software module can not recover on its own */
			ESP_LOG_WARN,       /*!< Error conditions from which recovery measures have been taken */
			ESP_LOG_INFO,       /*!< Information messages which describe normal flow of events */
			ESP_LOG_DEBUG,      /*!< Extra information which is not necessary for normal use (values, pointers, sizes, etc). */
			ESP_LOG_VERBOSE     /*!< Bigger chunks of debugging information, or frequent messages which can potentially flood the output. */
		} esp_log_level_t;

		extern unique_ptr<map<string, esp_log_level_t>> LOG_LEVELS_MAP;
		void esp_log_level_set(const char* tag, esp_log_level_t level);
		esp_log_level_t esp_log_level_get(const char* tag);
		#define ESP_LOGI(tag, _format, ...) { if(esp_log_level_get(tag) >= ESP_LOG_INFO) { printf("I "); printf(tag); printf(" "); printf(_format, ##__VA_ARGS__); } }
		#define ESP_LOGV(tag, _format, ...) { if(esp_log_level_get(tag) >= ESP_LOG_VERBOSE) { printf("V "); printf(tag); printf(" "); printf(_format, ##__VA_ARGS__); } }
		#define ESP_LOGD(tag, _format, ...) { if(esp_log_level_get(tag) >= ESP_LOG_DEBUG) { printf("D "); printf(tag); printf(" "); printf(_format, ##__VA_ARGS__); } }
		#define ESP_LOGE(tag, _format, ...) { if(esp_log_level_get(tag) >= ESP_LOG_ERROR) { printf("E "); printf(tag); printf(" "); printf(_format, ##__VA_ARGS__); } }
		#define ESP_LOGW(tag, _format, ...) { if(esp_log_level_get(tag) >= ESP_LOG_WARN) { printf("W "); printf(tag); printf(" "); printf(_format, ##__VA_ARGS__); } }

		void ESP_ERROR_CHECK(esp_err_t e);


		// FILESYSTEM FUNCTIONS

		typedef struct {
				const char* base_path;          /*!< File path prefix associated with the filesystem. */
				const char* partition_label;    /*!< Optional, label of SPIFFS partition to use. If set to NULL, first partition with subtype=spiffs will be used. */
				unsigned int max_files;         /*!< Maximum files that could be open at the same time. */
				bool format_if_mount_failed;    /*!< If true, it will format the file system if it fails to mount. */
		} esp_vfs_spiffs_conf_t;

		#define esp_vfs_spiffs_register(conf_ptr) ESP_OK
		#define esp_spiffs_info(l, t, u) ESP_OK
		#define esp_vfs_spiffs_unregister(ptr) ESP_OK

		
		// CPU/MEMORY FUNCTIONS

		/**
		 * @brief Flags to indicate the capabilities of the various memory systems
		 */
		#define MALLOC_CAP_EXEC             (1<<0)  ///< Memory must be able to run executable code
		#define MALLOC_CAP_32BIT            (1<<1)  ///< Memory must allow for aligned 32-bit data accesses
		#define MALLOC_CAP_8BIT             (1<<2)  ///< Memory must allow for 8/16/...-bit data accesses
		#define MALLOC_CAP_DMA              (1<<3)  ///< Memory must be able to accessed by DMA
		#define MALLOC_CAP_PID2             (1<<4)  ///< Memory must be mapped to PID2 memory space (PIDs are not currently used)
		#define MALLOC_CAP_PID3             (1<<5)  ///< Memory must be mapped to PID3 memory space (PIDs are not currently used)
		#define MALLOC_CAP_PID4             (1<<6)  ///< Memory must be mapped to PID4 memory space (PIDs are not currently used)
		#define MALLOC_CAP_PID5             (1<<7)  ///< Memory must be mapped to PID5 memory space (PIDs are not currently used)
		#define MALLOC_CAP_PID6             (1<<8)  ///< Memory must be mapped to PID6 memory space (PIDs are not currently used)
		#define MALLOC_CAP_PID7             (1<<9)  ///< Memory must be mapped to PID7 memory space (PIDs are not currently used)
		#define MALLOC_CAP_SPIRAM           (1<<10) ///< Memory must be in SPI RAM
		#define MALLOC_CAP_INTERNAL         (1<<11) ///< Memory must be internal; specifically it should not disappear when flash/spiram cache is switched off
		#define MALLOC_CAP_DEFAULT          (1<<12) ///< Memory can be returned in a non-capability-specific memory allocation (e.g. malloc(), calloc()) call
		#define MALLOC_CAP_IRAM_8BIT        (1<<13) ///< Memory must be in IRAM and allow unaligned access
		#define MALLOC_CAP_RETENTION        (1<<14)

		#define MALLOC_CAP_INVALID          (1<<31) ///< Memory can't be used / list end marker

		typedef struct multi_heap_info {
			unsigned long total_free_bytes;
			unsigned long total_allocated_bytes;
		} multi_heap_info_t;

		typedef struct rtc_cpu_freq_config {
			unsigned long freq_mhz;
		} rtc_cpu_freq_config_t;

		void heap_caps_get_info(multi_heap_info_t* info, uint32_t caps);

		void rtc_clk_cpu_freq_get_config(rtc_cpu_freq_config_t* info);
		void rtc_clk_cpu_freq_mhz_to_config(uint32_t mhz, rtc_cpu_freq_config_t* out);
		void rtc_clk_cpu_freq_set_config(rtc_cpu_freq_config_t* info);

		#define esp_get_free_heap_size() 320000
		size_t heap_caps_get_largest_free_block(uint32_t caps);



        //
		// WIFI AND NETWORKING OMITTED!
        //



		// TASKS AND TIME

		/** Class to handle the Thread Pool */
		class BriandIDFPortingTaskHandle {
			public:
			bool toBeKilled;
			std::thread::native_handle_type handle;
			std::thread::id thread_id;
			string name;

			BriandIDFPortingTaskHandle(const std::thread::native_handle_type& h, const char* name, const std::thread::id& tid);
			~BriandIDFPortingTaskHandle();
		};

		#define portTICK_PERIOD_MS 1

		typedef uint64_t TickType_t;
		typedef int BaseType_t;
		typedef uint16_t UBaseType_t;
		typedef void (*TaskFunction_t)( void * );

		/** Task states returned by eTaskGetState. */
		typedef enum
		{
			eRunning = 0,	/* A task is querying the state of itself, so must be running. */
			eReady,			/* The task being queried is in a read or pending ready list. */
			eBlocked,		/* The task being queried is in the Blocked state. */
			eSuspended,		/* The task being queried is in the Suspended state, or is in the Blocked state with an infinite time out. */
			eDeleted,		/* The task being queried has been deleted, but its TCB has not yet been freed. */
			eInvalid		/* Used as an 'invalid state' value. */
		} eTaskState;

		typedef unsigned char StackType_t;
		#define configSTACK_DEPTH_TYPE uint16_t
		typedef BriandIDFPortingTaskHandle* TaskHandle_t;

		/*
		*  Used with the uxTaskGetSystemState() function to return the state of each task in the system.
		*/
		typedef struct xTASK_STATUS
		{
			TaskHandle_t xHandle;			/* The handle of the task to which the rest of the information in the structure relates. */
			const char *pcTaskName;			/* A pointer to the task's name.  This value will be invalid if the task was deleted since the structure was populated! */ /*lint !e971 Unqualified char types are allowed for strings and single characters only. */
			UBaseType_t xTaskNumber;		/* A number unique to the task. */
			eTaskState eCurrentState;		/* The state in which the task existed when the structure was populated. */
			UBaseType_t uxCurrentPriority;	/* The priority at which the task was running (may be inherited) when the structure was populated. */
			UBaseType_t uxBasePriority;		/* The priority to which the task will return if the task's current priority has been inherited to avoid unbounded priority inversion when obtaining a mutex.  Only valid if configUSE_MUTEXES is defined as 1 in FreeRTOSConfig.h. */
			uint32_t ulRunTimeCounter;		/* The total run time allocated to the task so far, as defined by the run time stats clock.  See http://www.freertos.org/rtos-run-time-stats.html.  Only valid when configGENERATE_RUN_TIME_STATS is defined as 1 in FreeRTOSConfig.h. */
			StackType_t *pxStackBase;		/* Points to the lowest address of the task's stack area. */
			configSTACK_DEPTH_TYPE usStackHighWaterMark;	/* The minimum amount of stack space that has remained for the task since the task was created.  The closer this value is to zero the closer the task has come to overflowing its stack. */
		#if configTASKLIST_INCLUDE_COREID
			BaseType_t xCoreID;				/*!< Core this task is pinned to (0, 1, or -1 for tskNO_AFFINITY). This field is present if CONFIG_FREERTOS_VTASKLIST_INCLUDE_COREID is set. */
		#endif
		} TaskStatus_t;
		
		extern unique_ptr<vector<TaskHandle_t>> BRIAND_TASK_POOL;

		void vTaskDelay(TickType_t delay);

		uint64_t esp_timer_get_time();

		BaseType_t xTaskCreate(
				TaskFunction_t pvTaskCode,
				const char * const pcName,
				const uint32_t usStackDepth,
				void * const pvParameters,
				UBaseType_t uxPriority,
				TaskHandle_t * const pvCreatedTask);

		void vTaskDelete(TaskHandle_t handle);

		UBaseType_t uxTaskGetNumberOfTasks();
		UBaseType_t uxTaskGetSystemState( TaskStatus_t * const pxTaskStatusArray, const UBaseType_t uxArraySize, uint32_t * const pulTotalRunTime );


		// ESP PTHREADS

		/** pthread configuration structure that influences pthread creation */
		typedef struct {
			size_t stack_size;  ///< The stack size of the pthread
			size_t prio;        ///< The thread's priority
			bool inherit_cfg;   ///< Inherit this configuration further
			const char* thread_name;  ///< The thread name.
			int pin_to_core;    ///< The core id to pin the thread to. Has the same value range as xCoreId argument of xTaskCreatePinnedToCore.
		} esp_pthread_cfg_t;

		esp_pthread_cfg_t esp_pthread_get_default_config(void);
		esp_err_t esp_pthread_set_cfg(const esp_pthread_cfg_t *cfg);
		esp_err_t esp_pthread_get_cfg(esp_pthread_cfg_t *p);
		esp_err_t esp_pthread_init(void);
		

		// MISC

		esp_err_t nvs_flash_init(void);
		esp_err_t nvs_flash_erase(void);
		unsigned int esp_random();

    #endif

#endif