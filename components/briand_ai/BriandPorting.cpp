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

#if defined(__linux__) | defined(_WIN32)

    #include "BriandInclude.hxx"

	const char *esp_err_to_name(esp_err_t code) {
		return "UNDEFINED ON LINUX PLATFORM";
	}

	map<string, esp_log_level_t> LOG_LEVELS_MAP;
	
	void esp_log_level_set(const char* tag, esp_log_level_t level) {
		// If wildcard, all to level.
		if (strcmp(tag, "*") == 0) {
			for (auto it = LOG_LEVELS_MAP.begin(); it != LOG_LEVELS_MAP.end(); ++it) {
				it->second = level;
			}
		}
		else {
			LOG_LEVELS_MAP[string(tag)] = level;
		}
	}

	esp_log_level_t esp_log_level_get(const char* tag) {
		auto it = LOG_LEVELS_MAP.find(string(tag));
		
		if (it == LOG_LEVELS_MAP.end()) {
			// Create
			LOG_LEVELS_MAP[string(tag)] = ESP_LOG_NONE;
		}
		
		return LOG_LEVELS_MAP[string(tag)];
	}

	void ESP_ERROR_CHECK(esp_err_t e) { /* do nothing */ }

	void heap_caps_get_info(multi_heap_info_t* info, uint32_t caps) {
		bzero(info, sizeof(info));
		// Standard ESP 320KB
		// default return 0 for free bytes
		info->total_free_bytes = 0;
		info->total_allocated_bytes = 0;
	}

	void rtc_clk_cpu_freq_get_config(rtc_cpu_freq_config_t* info) { info->freq_mhz = 240; }
	void rtc_clk_cpu_freq_mhz_to_config(uint32_t mhz, rtc_cpu_freq_config_t* out) { out->freq_mhz = mhz; }
	void rtc_clk_cpu_freq_set_config(rtc_cpu_freq_config_t* info) { /* do nothing */ }

	size_t heap_caps_get_largest_free_block(uint32_t caps) { return 0; }

	BriandIDFPortingTaskHandle::BriandIDFPortingTaskHandle(const std::thread::native_handle_type& h, const char* name, const std::thread::id& tid) {
		this->handle = h;
		this->name = string(name);
		this->thread_id = tid;
		this->toBeKilled = false;
	}

	BriandIDFPortingTaskHandle::~BriandIDFPortingTaskHandle() {
		if (esp_log_level_get("ESPLinuxPorting") != ESP_LOG_NONE) cout << "BriandIDFPortingTaskHandle: " << this->name << " destroyed." << endl;
	}

	unique_ptr<vector<TaskHandle_t>> BRIAND_TASK_POOL = nullptr;

	TickType_t CTRL_C_MAX_WAIT = 0; // this is useful max waiting time before killing thread (see main()) 

	void vTaskDelay(TickType_t delay) { 
		if (CTRL_C_MAX_WAIT < delay) CTRL_C_MAX_WAIT = delay;
		std::this_thread::sleep_for( std::chrono::milliseconds(delay) ); 
	}

	uint64_t esp_timer_get_time() { 
		// Should return microseconds!
		auto clockPrecision = std::chrono::system_clock::now().time_since_epoch();
		auto micros = std::chrono::duration_cast<std::chrono::microseconds>(clockPrecision);
		return micros.count(); 
	}

	BaseType_t xTaskCreate(
			TaskFunction_t pvTaskCode,
			const char * const pcName,
			const uint32_t usStackDepth,
			void * const pvParameters,
			UBaseType_t uxPriority,
			TaskHandle_t * const pvCreatedTask)
	{
		// do not worry for prioriry and task depth now...

		std::thread t(pvTaskCode, pvParameters);
		TaskHandle_t tHandle = new BriandIDFPortingTaskHandle(t.native_handle(), pcName, t.get_id());

		if (pvCreatedTask != NULL) {
			*pvCreatedTask = tHandle;
		}

		// Add the task to pool BEFORE detach() otherwise native id is lost
		BRIAND_TASK_POOL->push_back( tHandle );

		t.detach(); // this will create daemon-like threads

		return static_cast<BaseType_t>(BRIAND_TASK_POOL->size()-1); // task index
	}

	void vTaskDelete(TaskHandle_t handle) {
		std::thread::id idToKill;

		if (handle == NULL || handle == nullptr) {
			// Terminate this
			idToKill = std::this_thread::get_id();
		}
		else {
			idToKill = handle->thread_id;
		}

		if (BRIAND_TASK_POOL != nullptr) {
			for (int i = 0; i<BRIAND_TASK_POOL->size(); i++) {
				if (BRIAND_TASK_POOL->at(i)->thread_id == idToKill) {
					BRIAND_TASK_POOL->at(i)->toBeKilled = true;
					break;
				}
			}
		}
	}

	UBaseType_t uxTaskGetNumberOfTasks() {
		if (BRIAND_TASK_POOL == nullptr) return 0;
		return static_cast<UBaseType_t>(BRIAND_TASK_POOL->size());
	}

	UBaseType_t uxTaskGetSystemState( TaskStatus_t * const pxTaskStatusArray, const UBaseType_t uxArraySize, uint32_t * const pulTotalRunTime ) {
		UBaseType_t max = 0;
		if (uxArraySize == 0) return 0;
		if (pulTotalRunTime != NULL) *pulTotalRunTime = 0;
		if (BRIAND_TASK_POOL != nullptr && pxTaskStatusArray != NULL) {
			max = (uxArraySize < static_cast<UBaseType_t>(BRIAND_TASK_POOL->size()) ? uxArraySize :  static_cast<UBaseType_t>(BRIAND_TASK_POOL->size()));
			for (unsigned short i=0; i<max; i++) {
				bzero(&pxTaskStatusArray[i], sizeof(TaskStatus_t));
				pxTaskStatusArray[i].xTaskNumber = i;
				pxTaskStatusArray[i].pcTaskName = BRIAND_TASK_POOL->at(i)->name.c_str();
				//
				// TODO : calculate phtread stack size
				//
				pxTaskStatusArray[i].usStackHighWaterMark = 0;
			}
		}
		return max;
	}

	esp_err_t nvs_flash_init(void) { return ESP_OK; }
	esp_err_t nvs_flash_erase(void) { return ESP_OK; }
	unsigned int esp_random() {
		return static_cast<unsigned int>(rand());
	}

	esp_pthread_cfg_t esp_pthread_get_default_config(void) {
		// Like the default configuration
		esp_pthread_cfg_t defaults;
		defaults.stack_size = 2048;
		defaults.inherit_cfg = false;
		defaults.pin_to_core = 0;
		defaults.prio = 5;
		defaults.thread_name = "pthread";
		return defaults;
	}

	esp_err_t esp_pthread_set_cfg(const esp_pthread_cfg_t *cfg) {
		// do nothing
		return ESP_OK;
	}

	esp_err_t esp_pthread_get_cfg(esp_pthread_cfg_t *p) {
		if (p != NULL) *p = esp_pthread_get_default_config();
		return ESP_OK;
	}

	esp_err_t esp_pthread_init(void) {
		// do nothing
		return ESP_OK;
	}

	// app_main() early declaration with extern keyword so will be found
	extern "C" { void app_main(); }

	// Ctrl-C event handler
	bool CTRL_C_EVENT_SET = false;
	void sig_hnd_Ctrl_C(int s) { CTRL_C_EVENT_SET = true; } 

	// main() method required

	int main(int argc, char** argv) {
		// srand for esp_random()
		srand(time(NULL));

		// Add this to the logging utils in order to deactivate output if necessary
		esp_log_level_set("ESPLinuxPorting", ESP_LOG_NONE);

		// Save this thread id
		cout << "MAIN THREAD ID: " << std::this_thread::get_id() << endl;

		// Attach Ctrl-C event handler
		sighandler_t oldHandler = signal(SIGINT, sig_hnd_Ctrl_C);

		// Will create the app_main() method and then remains waiting like esp
		// Will also do the task scheduler work to check if any thread should be killed
		cout << "main(): Starting. Creating task pool simulation..." << endl;

		BRIAND_TASK_POOL = make_unique<vector<TaskHandle_t>>();
		
		cout << "main() Pool started. Use Ctrl-C to terminate" << endl;
		
		cout << "Starting app_main()" << endl;

		app_main(); // This must not be a thread because it terminates!

		cout << "app_main() started." << endl;

		while(!CTRL_C_EVENT_SET) { 
			// Check if any instanced thread should be terminated
			for (int i=0; i<BRIAND_TASK_POOL->size(); i++) {
				if (BRIAND_TASK_POOL->at(i)->toBeKilled) {
					string tname = BRIAND_TASK_POOL->at(i)->name;
					pthread_cancel(BRIAND_TASK_POOL->at(i)->handle);
					delete BRIAND_TASK_POOL->at(i);
					BRIAND_TASK_POOL->erase(BRIAND_TASK_POOL->begin() + i);
					if (esp_log_level_get("ESPLinuxPorting") != ESP_LOG_NONE) cout << "Thread #" << i << "(" << tname << ") killed" << endl;
				}
			}
				
			std::this_thread::sleep_for( std::chrono::milliseconds(500) ); 
		}

		cout << endl << endl << "*** Ctrl-C event caught! ***" << endl << endl;

		// Reset the original signal handler
		signal(SIGINT, oldHandler);

		// Kill all processes (from newer to older)
		for (int i=BRIAND_TASK_POOL->size() - 1; i>=0; i--) {
			if (BRIAND_TASK_POOL->at(i) != NULL) {
				string tname = BRIAND_TASK_POOL->at(i)->name;
				pthread_cancel(BRIAND_TASK_POOL->at(i)->handle);
				delete BRIAND_TASK_POOL->at(i);
				cout << "Thread #" << i << "(" << tname << ") killed" << endl;
			} 
		}
				
		cout << endl << endl << "*** All threads killed! Exiting. ***" << endl << endl;
		raise(SIGINT);

		return 0;
	}

#endif