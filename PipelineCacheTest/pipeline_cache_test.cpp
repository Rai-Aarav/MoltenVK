// Exercises the MoltenVK pipeline cache: retrieving cache data, reloading it, and creating
// pipelines from several threads that share one pipeline cache. Links against libMoltenVK directly.
//
// Usage: pipeline_cache_test [shaderCount] [threadCount] [save|load <file>]
//
// Each run perturbs the shaders with a new seed, so that neither Metal's own shader cache,
// nor results left behind by an earlier run, affect the timings.

#include <vulkan/vulkan.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "shader_spv.h"

static const uint32_t kSeedPlaceholder = 0x5EEDC0DE;
static const uint32_t kElementCount = 16;

static int failureCount = 0;

#define CHECK(cond, ...) do { \
	if (cond) { printf("  PASS: "); } else { printf("  FAIL: "); failureCount++; } \
	printf(__VA_ARGS__); printf("\n"); \
} while (0)

#define REQUIRE(cond, msg) do { if ( !(cond) ) { printf("FATAL: %s\n", msg); exit(2); } } while (0)

static double now() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Same hash as the shader, so the shader output can be verified.
static uint32_t shaderHash(uint32_t x) {
	x ^= x >> 16; x *= 0x7feb352du;
	x ^= x >> 15; x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

struct Context {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t queueFamily = 0;
	VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	uint32_t* mapped = nullptr;
	VkCommandPool commandPool = VK_NULL_HANDLE;
};

static void initContext(Context& ctx) {
	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.apiVersion = VK_API_VERSION_1_1;
	VkInstanceCreateInfo instInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	instInfo.pApplicationInfo = &appInfo;
	REQUIRE(vkCreateInstance(&instInfo, nullptr, &ctx.instance) == VK_SUCCESS, "vkCreateInstance");

	uint32_t pdCount = 1;
	VkResult rslt = vkEnumeratePhysicalDevices(ctx.instance, &pdCount, &ctx.physicalDevice);
	REQUIRE((rslt == VK_SUCCESS || rslt == VK_INCOMPLETE) && pdCount == 1, "vkEnumeratePhysicalDevices");

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
	printf("Device: %s\n", props.deviceName);

	uint32_t qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qfCount, nullptr);
	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice, &qfCount, qfProps.data());
	for (uint32_t i = 0; i < qfCount; i++) {
		if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.queueFamily = i; break; }
	}

	float priority = 1.0f;
	VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueInfo.queueFamilyIndex = ctx.queueFamily;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;
	VkDeviceCreateInfo devInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	devInfo.queueCreateInfoCount = 1;
	devInfo.pQueueCreateInfos = &queueInfo;
	REQUIRE(vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device) == VK_SUCCESS, "vkCreateDevice");
	vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

	VkDescriptorSetLayoutBinding binding = {};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	VkDescriptorSetLayoutCreateInfo dslInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	dslInfo.bindingCount = 1;
	dslInfo.pBindings = &binding;
	REQUIRE(vkCreateDescriptorSetLayout(ctx.device, &dslInfo, nullptr, &ctx.setLayout) == VK_SUCCESS, "vkCreateDescriptorSetLayout");

	VkPipelineLayoutCreateInfo plInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	plInfo.setLayoutCount = 1;
	plInfo.pSetLayouts = &ctx.setLayout;
	REQUIRE(vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &ctx.pipelineLayout) == VK_SUCCESS, "vkCreatePipelineLayout");

	// A host-visible storage buffer for the shader to write its results to.
	VkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufInfo.size = kElementCount * sizeof(uint32_t);
	bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	REQUIRE(vkCreateBuffer(ctx.device, &bufInfo, nullptr, &ctx.buffer) == VK_SUCCESS, "vkCreateBuffer");
	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(ctx.device, ctx.buffer, &memReqs);
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memProps);
	uint32_t memType = UINT32_MAX;
	VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
		if ((memReqs.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & wanted) == wanted) { memType = i; break; }
	}
	REQUIRE(memType != UINT32_MAX, "no host-visible memory type");
	VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = memType;
	REQUIRE(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &ctx.memory) == VK_SUCCESS, "vkAllocateMemory");
	REQUIRE(vkBindBufferMemory(ctx.device, ctx.buffer, ctx.memory, 0) == VK_SUCCESS, "vkBindBufferMemory");
	REQUIRE(vkMapMemory(ctx.device, ctx.memory, 0, VK_WHOLE_SIZE, 0, (void**)&ctx.mapped) == VK_SUCCESS, "vkMapMemory");

	VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
	VkDescriptorPoolCreateInfo dpInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	dpInfo.maxSets = 1;
	dpInfo.poolSizeCount = 1;
	dpInfo.pPoolSizes = &poolSize;
	REQUIRE(vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &ctx.descriptorPool) == VK_SUCCESS, "vkCreateDescriptorPool");
	VkDescriptorSetAllocateInfo dsInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	dsInfo.descriptorPool = ctx.descriptorPool;
	dsInfo.descriptorSetCount = 1;
	dsInfo.pSetLayouts = &ctx.setLayout;
	REQUIRE(vkAllocateDescriptorSets(ctx.device, &dsInfo, &ctx.descriptorSet) == VK_SUCCESS, "vkAllocateDescriptorSets");
	VkDescriptorBufferInfo dbInfo = { ctx.buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	write.dstSet = ctx.descriptorSet;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &dbInfo;
	vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

	VkCommandPoolCreateInfo cpInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	cpInfo.queueFamilyIndex = ctx.queueFamily;
	REQUIRE(vkCreateCommandPool(ctx.device, &cpInfo, nullptr, &ctx.commandPool) == VK_SUCCESS, "vkCreateCommandPool");
}

// Creates a set of shader modules that differ from each other, and from those of any other run.
static std::vector<VkShaderModule> createShaderModules(Context& ctx, uint32_t count, std::vector<uint32_t>& seeds,
													   const uint32_t* pBaseSeed = nullptr) {
	std::random_device rd;
	std::vector<VkShaderModule> modules(count);
	seeds.resize(count);
	std::vector<uint32_t> code(shader_spv, shader_spv + shader_spv_word_count);
	std::vector<size_t> seedPositions;
	for (size_t i = 5; i < code.size(); i++) {
		if (code[i] == kSeedPlaceholder) { seedPositions.push_back(i); }
	}
	REQUIRE(seedPositions.size() == 1, "expected exactly one seed constant in the SPIR-V");

	uint32_t baseSeed = pBaseSeed ? *pBaseSeed : rd();
	for (uint32_t i = 0; i < count; i++) {
		seeds[i] = baseSeed + i * 7919u;
		code[seedPositions[0]] = seeds[i];
		VkShaderModuleCreateInfo smInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		smInfo.codeSize = code.size() * sizeof(uint32_t);
		smInfo.pCode = code.data();
		REQUIRE(vkCreateShaderModule(ctx.device, &smInfo, nullptr, &modules[i]) == VK_SUCCESS, "vkCreateShaderModule");
	}
	return modules;
}

struct PipelineBatchResult {
	std::vector<VkPipeline> pipelines;
	double seconds = 0;
	uint32_t failures = 0;
	uint32_t stageCacheHits = 0;
};

// Creates one compute pipeline per shader module, using the specified number of threads, all sharing the
// one pipeline cache. If sameShadersOnAllThreads is true, every thread creates a pipeline for every module.
static PipelineBatchResult createPipelines(Context& ctx, VkPipelineCache cache,
										   const std::vector<VkShaderModule>& modules,
										   uint32_t threadCount, bool sameShadersOnAllThreads = false) {
	PipelineBatchResult result;
	size_t slotCount = sameShadersOnAllThreads ? modules.size() * threadCount : modules.size();
	result.pipelines.assign(slotCount, VK_NULL_HANDLE);
	std::atomic<uint32_t> failures(0), cacheHits(0), nextIndex(0);

	auto createOne = [&](size_t moduleIdx, size_t slot) {
		VkPipelineCreationFeedback pipelineFeedback = {};
		VkPipelineCreationFeedback stageFeedback = {};
		VkPipelineCreationFeedbackCreateInfo feedbackInfo = { VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO };
		feedbackInfo.pPipelineCreationFeedback = &pipelineFeedback;
		feedbackInfo.pipelineStageCreationFeedbackCount = 1;
		feedbackInfo.pPipelineStageCreationFeedbacks = &stageFeedback;

		VkComputePipelineCreateInfo cpInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		cpInfo.pNext = &feedbackInfo;
		cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		cpInfo.stage.module = modules[moduleIdx];
		cpInfo.stage.pName = "main";
		cpInfo.layout = ctx.pipelineLayout;
		VkResult rslt = vkCreateComputePipelines(ctx.device, cache, 1, &cpInfo, nullptr, &result.pipelines[slot]);
		if (rslt != VK_SUCCESS || result.pipelines[slot] == VK_NULL_HANDLE) { failures++; }
		if (stageFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) { cacheHits++; }
	};

	double start = now();
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < threadCount; t++) {
		threads.emplace_back([&, t]() {
			if (sameShadersOnAllThreads) {
				for (size_t i = 0; i < modules.size(); i++) { createOne(i, t * modules.size() + i); }
			} else {
				for (uint32_t i; (i = nextIndex++) < modules.size(); ) { createOne(i, i); }
			}
		});
	}
	for (auto& th : threads) { th.join(); }
	result.seconds = now() - start;
	result.failures = failures;
	result.stageCacheHits = cacheHits;
	return result;
}

static void destroyPipelines(Context& ctx, PipelineBatchResult& batch) {
	for (auto p : batch.pipelines) { vkDestroyPipeline(ctx.device, p, nullptr); }
	batch.pipelines.clear();
}

static void destroyShaderModules(Context& ctx, std::vector<VkShaderModule>& modules) {
	for (auto m : modules) { vkDestroyShaderModule(ctx.device, m, nullptr); }
	modules.clear();
}

static VkPipelineCache createCache(Context& ctx, const void* data = nullptr, size_t size = 0, VkResult* pResult = nullptr) {
	VkPipelineCacheCreateInfo pcInfo = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
	pcInfo.initialDataSize = size;
	pcInfo.pInitialData = data;
	VkPipelineCache cache = VK_NULL_HANDLE;
	VkResult rslt = vkCreatePipelineCache(ctx.device, &pcInfo, nullptr, &cache);
	if (pResult) { *pResult = rslt; } else { REQUIRE(rslt == VK_SUCCESS, "vkCreatePipelineCache"); }
	return cache;
}

// Runs the pipeline and returns whether the shader wrote the values expected for its seed.
static bool runAndVerify(Context& ctx, VkPipeline pipeline, uint32_t seed) {
	memset(ctx.mapped, 0, kElementCount * sizeof(uint32_t));

	VkCommandBufferAllocateInfo cbInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cbInfo.commandPool = ctx.commandPool;
	cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbInfo.commandBufferCount = 1;
	VkCommandBuffer cmd;
	if (vkAllocateCommandBuffers(ctx.device, &cbInfo, &cmd) != VK_SUCCESS) { return false; }
	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmd, &beginInfo);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipelineLayout, 0, 1, &ctx.descriptorSet, 0, nullptr);
	vkCmdDispatch(cmd, kElementCount, 1, 1);
	vkEndCommandBuffer(cmd);
	VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	bool ok = (vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS) && (vkQueueWaitIdle(ctx.queue) == VK_SUCCESS);
	vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &cmd);

	for (uint32_t i = 0; ok && i < kElementCount; i++) {
		if (ctx.mapped[i] != shaderHash(seed + i)) { ok = false; }
	}
	return ok;
}

// Populates a pipeline cache in this process, and saves its data, along with the seed of the shaders, to a file.
static int saveCacheFile(Context& ctx, uint32_t shaderCount, uint32_t threadCount, const char* path) {
	std::random_device rd;
	uint32_t baseSeed = rd();
	std::vector<uint32_t> seeds;
	auto modules = createShaderModules(ctx, shaderCount, seeds, &baseSeed);
	VkPipelineCache cache = createCache(ctx);
	auto batch = createPipelines(ctx, cache, modules, threadCount);
	CHECK(batch.failures == 0, "%u threads: %u pipelines in %.2f s", threadCount, shaderCount, batch.seconds);
	printf("RESULT populate_seconds %.2f\n", batch.seconds);

	size_t size = 0;
	vkGetPipelineCacheData(ctx.device, cache, &size, nullptr);
	std::vector<uint8_t> data(size);
	VkResult rslt = vkGetPipelineCacheData(ctx.device, cache, &size, data.data());
	CHECK(rslt == VK_SUCCESS && size == data.size(), "retrieved %zu bytes of cache data", size);

	FILE* f = fopen(path, "wb");
	REQUIRE(f, "could not open the cache file for writing");
	fwrite(&baseSeed, sizeof(baseSeed), 1, f);
	fwrite(&shaderCount, sizeof(shaderCount), 1, f);
	fwrite(data.data(), 1, data.size(), f);
	fclose(f);
	printf("RESULT failures %d\n", failureCount);
	return failureCount ? 1 : 0;
}

// Loads pipeline cache data that was saved by an earlier process, as an app does when it is restarted,
// and creates pipelines for the first usedCount of the shaders in it.
static int loadCacheFile(Context& ctx, uint32_t usedCount, uint32_t threadCount, const char* path) {
	FILE* f = fopen(path, "rb");
	REQUIRE(f, "could not open the cache file for reading");
	uint32_t baseSeed = 0, shaderCount = 0;
	REQUIRE(fread(&baseSeed, sizeof(baseSeed), 1, f) == 1 && fread(&shaderCount, sizeof(shaderCount), 1, f) == 1, "cache file header");
	std::vector<uint8_t> data;
	uint8_t chunk[65536];
	for (size_t n; (n = fread(chunk, 1, sizeof(chunk), f)) > 0; ) { data.insert(data.end(), chunk, chunk + n); }
	fclose(f);
	if (usedCount == 0 || usedCount > shaderCount) { usedCount = shaderCount; }

	std::vector<uint32_t> seeds;
	auto modules = createShaderModules(ctx, usedCount, seeds, &baseSeed);

	double start = now();
	VkPipelineCache cache = createCache(ctx, data.data(), data.size());
	double loadSeconds = now() - start;
	printf("  INFO: vkCreatePipelineCache with %zu bytes for %u shaders took %.3f s\n", data.size(), shaderCount, loadSeconds);

	auto batch = createPipelines(ctx, cache, modules, threadCount);
	CHECK(batch.failures == 0, "%u threads: %u pipelines from the loaded cache in %.2f s", threadCount, usedCount, batch.seconds);
	CHECK(batch.stageCacheHits == usedCount, "all shaders reported as pipeline cache hits (%u of %u)", batch.stageCacheHits, usedCount);
	bool allCorrect = true;
	for (uint32_t i = 0; i < usedCount; i += (usedCount / 8 ? usedCount / 8 : 1)) {
		allCorrect = allCorrect && runAndVerify(ctx, batch.pipelines[i], seeds[i]);
	}
	CHECK(allCorrect, "pipelines created from the loaded cache produce correct results");

	size_t size = 0;
	vkGetPipelineCacheData(ctx.device, cache, &size, nullptr);
	CHECK(size == data.size(), "loaded cache saves the same amount of data (%zu bytes)", size);

	printf("RESULT cache_load_seconds %.3f\n", loadSeconds);
	printf("RESULT cached_pipelines_seconds %.3f\n", batch.seconds);
	printf("RESULT cache_load_plus_pipelines_seconds %.3f\n", loadSeconds + batch.seconds);
	printf("RESULT failures %d\n", failureCount);
	return failureCount ? 1 : 0;
}

int main(int argc, char** argv) {
	uint32_t shaderCount = argc > 1 ? (uint32_t)atoi(argv[1]) : 192;
	uint32_t threadCount = argc > 2 ? (uint32_t)atoi(argv[2]) : 8;
	setvbuf(stdout, nullptr, _IOLBF, 0);

	Context ctx;
	initContext(ctx);
	printf("Shaders: %u, threads: %u\n", shaderCount, threadCount);

	// pipeline_cache_test <shaderCount> <threadCount> save <file>
	// pipeline_cache_test <usedShaderCount, or 0 for all> <threadCount> load <file>
	if (argc > 4 && strcmp(argv[3], "save") == 0) { return saveCacheFile(ctx, shaderCount, threadCount, argv[4]); }
	if (argc > 4 && strcmp(argv[3], "load") == 0) { return loadCacheFile(ctx, shaderCount, threadCount, argv[4]); }

	std::vector<uint8_t> cacheData;

	printf("\n[1] Retrieve pipeline cache data without first querying its size\n");
	{
		std::vector<uint32_t> seeds;
		auto modules = createShaderModules(ctx, 8, seeds);
		VkPipelineCache cache = createCache(ctx);
		auto batch = createPipelines(ctx, cache, modules, 1);
		CHECK(batch.failures == 0, "created %zu pipelines", modules.size());

		std::vector<uint8_t> direct(16 * 1024 * 1024);
		size_t directSize = direct.size();
		VkResult rslt = vkGetPipelineCacheData(ctx.device, cache, &directSize, direct.data());
		CHECK(rslt == VK_SUCCESS && directSize > 64 && directSize < direct.size(),
			  "data retrieved into a large buffer with no size query (result %d, %zu bytes)", rslt, directSize);

		size_t querySize = 0;
		rslt = vkGetPipelineCacheData(ctx.device, cache, &querySize, nullptr);
		CHECK(rslt == VK_SUCCESS && querySize == directSize, "size query agrees (%zu bytes)", querySize);

		std::vector<uint8_t> exact(querySize);
		size_t exactSize = querySize;
		rslt = vkGetPipelineCacheData(ctx.device, cache, &exactSize, exact.data());
		CHECK(rslt == VK_SUCCESS && exactSize == querySize && directSize == querySize && memcmp(exact.data(), direct.data(), querySize) == 0,
			  "data retrieved into an exact size buffer is identical");

		// Query the size, then let the cache grow before retrieving the data, as happens when
		// another thread creates a pipeline between the two calls.
		size_t staleSize = 0;
		vkGetPipelineCacheData(ctx.device, cache, &staleSize, nullptr);
		std::vector<uint32_t> moreSeeds;
		auto moreModules = createShaderModules(ctx, 2, moreSeeds);
		auto moreBatch = createPipelines(ctx, cache, moreModules, 1);
		std::vector<uint8_t> roomy(staleSize * 4);
		size_t roomySize = roomy.size();
		rslt = vkGetPipelineCacheData(ctx.device, cache, &roomySize, roomy.data());
		CHECK(rslt == VK_SUCCESS && roomySize > staleSize,
			  "data retrieved after the cache grew since the size query (result %d, %zu bytes, was %zu)", rslt, roomySize, staleSize);

		size_t smallSize = staleSize;
		std::vector<uint8_t> small(smallSize);
		rslt = vkGetPipelineCacheData(ctx.device, cache, &smallSize, small.data());
		CHECK(rslt == VK_INCOMPLETE, "buffer that is now too small reports VK_INCOMPLETE (result %d)", rslt);

		destroyPipelines(ctx, batch);
		destroyPipelines(ctx, moreBatch);
		destroyShaderModules(ctx, modules);
		destroyShaderModules(ctx, moreModules);
		vkDestroyPipelineCache(ctx.device, cache, nullptr);
	}

	std::vector<uint32_t> seeds;
	auto modules = createShaderModules(ctx, shaderCount, seeds);

	printf("\n[2] Create pipelines from %u threads sharing one pipeline cache\n", threadCount);
	{
		// A separate set of equally sized shaders gives the single-threaded time to compare against.
		std::vector<uint32_t> refSeeds;
		auto refModules = createShaderModules(ctx, shaderCount, refSeeds);
		VkPipelineCache refCache = createCache(ctx);
		auto refBatch = createPipelines(ctx, refCache, refModules, 1);
		CHECK(refBatch.failures == 0, "1 thread:  %u pipelines in %.2f s", shaderCount, refBatch.seconds);
		destroyPipelines(ctx, refBatch);
		destroyShaderModules(ctx, refModules);
		vkDestroyPipelineCache(ctx.device, refCache, nullptr);

		VkPipelineCache cache = createCache(ctx);
		auto batch = createPipelines(ctx, cache, modules, threadCount);
		CHECK(batch.failures == 0, "%u threads: %u pipelines in %.2f s", threadCount, shaderCount, batch.seconds);
		CHECK(batch.stageCacheHits == 0, "no cache hits reported for shaders that are new to the cache (%u)", batch.stageCacheHits);
		printf("  INFO: speedup from %u threads: %.2fx\n", threadCount, refBatch.seconds / batch.seconds);
		printf("RESULT concurrent_speedup %.2f\n", refBatch.seconds / batch.seconds);
		printf("RESULT single_thread_seconds %.2f\n", refBatch.seconds);
		printf("RESULT multi_thread_seconds %.2f\n", batch.seconds);

		bool allCorrect = true;
		for (uint32_t i = 0; i < shaderCount; i += (shaderCount / 8 ? shaderCount / 8 : 1)) {
			allCorrect = allCorrect && runAndVerify(ctx, batch.pipelines[i], seeds[i]);
		}
		CHECK(allCorrect, "pipelines created concurrently produce correct results");

		size_t size = 0;
		vkGetPipelineCacheData(ctx.device, cache, &size, nullptr);
		cacheData.resize(size);
		VkResult rslt = vkGetPipelineCacheData(ctx.device, cache, &size, cacheData.data());
		CHECK(rslt == VK_SUCCESS && size == cacheData.size(), "saved %zu bytes of cache data for %u shaders", size, shaderCount);

		destroyPipelines(ctx, batch);
		vkDestroyPipelineCache(ctx.device, cache, nullptr);
	}

	printf("\n[3] Every thread creates pipelines for the same shaders at the same time\n");
	{
		std::vector<uint32_t> sharedSeeds;
		auto sharedModules = createShaderModules(ctx, 16, sharedSeeds);
		VkPipelineCache cache = createCache(ctx);
		auto batch = createPipelines(ctx, cache, sharedModules, threadCount, true);
		CHECK(batch.failures == 0, "%zu pipelines created for %zu shaders", batch.pipelines.size(), sharedModules.size());
		bool allCorrect = true;
		for (uint32_t t = 0; t < threadCount; t++) {
			size_t slot = t * sharedModules.size() + (t % sharedModules.size());
			allCorrect = allCorrect && runAndVerify(ctx, batch.pipelines[slot], sharedSeeds[t % sharedModules.size()]);
		}
		CHECK(allCorrect, "pipelines produce correct results");

		// Each shader should have been compiled, and so added to the cache, exactly once.
		size_t sharedSize = 0, singleSize = 0;
		vkGetPipelineCacheData(ctx.device, cache, &sharedSize, nullptr);
		VkPipelineCache singleCache = createCache(ctx);
		auto singleBatch = createPipelines(ctx, singleCache, sharedModules, 1);
		vkGetPipelineCacheData(ctx.device, singleCache, &singleSize, nullptr);
		CHECK(sharedSize == singleSize, "cache holds each shader once (%zu bytes, %zu when created on one thread)", sharedSize, singleSize);

		destroyPipelines(ctx, batch);
		destroyPipelines(ctx, singleBatch);
		destroyShaderModules(ctx, sharedModules);
		vkDestroyPipelineCache(ctx.device, cache, nullptr);
		vkDestroyPipelineCache(ctx.device, singleCache, nullptr);
	}

	printf("\n[4] Reload the saved pipeline cache data\n");
	{
		double start = now();
		VkPipelineCache cache = createCache(ctx, cacheData.data(), cacheData.size());
		double loadSeconds = now() - start;
		printf("  INFO: vkCreatePipelineCache with %zu bytes for %u shaders took %.3f s\n", cacheData.size(), shaderCount, loadSeconds);
		printf("RESULT cache_load_seconds %.3f\n", loadSeconds);

		// Data loaded but never used must be written back out unchanged.
		size_t size = 0;
		vkGetPipelineCacheData(ctx.device, cache, &size, nullptr);
		std::vector<uint8_t> resaved(size);
		VkResult rslt = vkGetPipelineCacheData(ctx.device, cache, &size, resaved.data());
		CHECK(rslt == VK_SUCCESS && size == cacheData.size(), "reloaded cache saves the same amount of data before use (%zu bytes)", size);

		auto batch = createPipelines(ctx, cache, modules, threadCount);
		CHECK(batch.failures == 0, "%u threads: %u pipelines from the reloaded cache in %.2f s", threadCount, shaderCount, batch.seconds);
		CHECK(batch.stageCacheHits == shaderCount, "all shaders reported as pipeline cache hits (%u of %u)", batch.stageCacheHits, shaderCount);
		printf("RESULT cached_pipelines_seconds %.2f\n", batch.seconds);
		printf("RESULT cache_load_plus_pipelines_seconds %.2f\n", loadSeconds + batch.seconds);

		bool allCorrect = true;
		for (uint32_t i = 0; i < shaderCount; i += (shaderCount / 8 ? shaderCount / 8 : 1)) {
			allCorrect = allCorrect && runAndVerify(ctx, batch.pipelines[i], seeds[i]);
		}
		CHECK(allCorrect, "pipelines created from the reloaded cache produce correct results");

		size = 0;
		vkGetPipelineCacheData(ctx.device, cache, &size, nullptr);
		CHECK(size == cacheData.size(), "reloaded cache saves the same amount of data after use (%zu bytes)", size);

		destroyPipelines(ctx, batch);
		vkDestroyPipelineCache(ctx.device, cache, nullptr);
	}

	printf("\n[5] Merge a reloaded pipeline cache into another before either is used\n");
	{
		VkPipelineCache src = createCache(ctx, cacheData.data(), cacheData.size());
		VkPipelineCache dst = createCache(ctx);
		VkResult rslt = vkMergePipelineCaches(ctx.device, dst, 1, &src);
		CHECK(rslt == VK_SUCCESS, "vkMergePipelineCaches (result %d)", rslt);
		vkDestroyPipelineCache(ctx.device, src, nullptr);

		std::vector<VkShaderModule> subset(modules.begin(), modules.begin() + (shaderCount < 16 ? shaderCount : 16));
		auto batch = createPipelines(ctx, dst, subset, threadCount);
		CHECK(batch.failures == 0 && batch.stageCacheHits == subset.size(),
			  "%zu pipelines from the merged cache, %u cache hits", subset.size(), batch.stageCacheHits);
		CHECK(runAndVerify(ctx, batch.pipelines[0], seeds[0]), "pipeline from the merged cache produces correct results");
		size_t size = 0;
		vkGetPipelineCacheData(ctx.device, dst, &size, nullptr);
		CHECK(size == cacheData.size(), "merged cache saves the same amount of data (%zu bytes)", size);
		destroyPipelines(ctx, batch);
		vkDestroyPipelineCache(ctx.device, dst, nullptr);
	}

	printf("\n[6] Load truncated and corrupt pipeline cache data\n");
	{
		VkResult rslt;
		size_t truncatedSize = cacheData.size() * 6 / 10;
		VkPipelineCache cache = createCache(ctx, cacheData.data(), truncatedSize, &rslt);
		CHECK(rslt == VK_SUCCESS && cache != VK_NULL_HANDLE, "truncated data does not fail vkCreatePipelineCache (result %d)", rslt);
		if (cache != VK_NULL_HANDLE) {
			auto batch = createPipelines(ctx, cache, modules, threadCount);
			CHECK(batch.failures == 0, "all %u pipelines created from the truncated cache, %u cache hits", shaderCount, batch.stageCacheHits);
			CHECK(batch.stageCacheHits > 0 && batch.stageCacheHits < shaderCount, "the intact part of the truncated data was used");
			CHECK(runAndVerify(ctx, batch.pipelines[shaderCount - 1], seeds[shaderCount - 1]), "pipeline produces correct results");
			destroyPipelines(ctx, batch);
			vkDestroyPipelineCache(ctx.device, cache, nullptr);
		}

		// Keep the header, which is all the app can validate, and replace the content.
		std::vector<uint8_t> garbage(cacheData.begin(), cacheData.begin() + 4096);
		for (size_t i = 32; i < garbage.size(); i++) { garbage[i] = (uint8_t)(i * 31 + 7); }
		cache = createCache(ctx, garbage.data(), garbage.size(), &rslt);
		CHECK(rslt == VK_SUCCESS && cache != VK_NULL_HANDLE, "corrupt data does not fail vkCreatePipelineCache (result %d)", rslt);
		if (cache != VK_NULL_HANDLE) { vkDestroyPipelineCache(ctx.device, cache, nullptr); }
	}

	destroyShaderModules(ctx, modules);
	vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
	vkDestroyDescriptorPool(ctx.device, ctx.descriptorPool, nullptr);
	vkDestroyBuffer(ctx.device, ctx.buffer, nullptr);
	vkFreeMemory(ctx.device, ctx.memory, nullptr);
	vkDestroyPipelineLayout(ctx.device, ctx.pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, ctx.setLayout, nullptr);
	vkDestroyDevice(ctx.device, nullptr);
	vkDestroyInstance(ctx.instance, nullptr);

	printf("\n%s (%d failed)\n", failureCount ? "FAILED" : "PASSED", failureCount);
	printf("RESULT failures %d\n", failureCount);
	return failureCount ? 1 : 0;
}
