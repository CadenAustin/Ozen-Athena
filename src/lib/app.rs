use anyhow::{anyhow, Result};
use cgmath::{point3, vec3, Deg};
use std::{mem::size_of, ptr::copy_nonoverlapping as memcpy, time::Instant};
use winit::window::Window;

use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_0::*,
    vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension},
    window as vk_window, Version,
};

use crate::{
    command_buffer::{create_command_buffers, create_command_pools},
    depth_object::create_depth_objects,
    descriptor_layout::create_description_set_layout,
    descriptor_pool::{create_descriptor_pool, create_descriptor_sets},
    framebuffer::create_framebuffers,
    image::create_color_objects,
    instance::create_instance,
    logical_device::create_logical_device,
    model::load_model,
    physical_device::pick_physical_device,
    pipeline::create_pipeline,
    render_pass::create_render_pass,
    swapchain::{create_swapchain, create_swapchain_image_views},
    sync_objects::create_sync_objects,
    texture::{create_texture_image, create_texture_image_view, create_texture_sampler},
    types::Mat4,
    uniform_buffer::{create_uniform_buffers, UniformBufferObject},
    vertex::Vertex,
    vertex_buffer::{create_index_buffer, create_vertex_buffer},
};

pub(crate) const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
pub(crate) const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub(crate) const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
pub(crate) const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone, Debug)]
pub struct App {
    _entry: Entry,
    instance: Instance,
    data: AppData,
    pub device: Device,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub models: usize,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY).unwrap();
        let _entry = Entry::new(loader).map_err(|b| anyhow!("{}", b)).unwrap();
        let mut data = AppData::default();
        let instance = create_instance(window, &_entry, &mut data).unwrap();
        data.surface = vk_window::create_surface(&instance, &window, &window).unwrap();
        pick_physical_device(&instance, &mut data).unwrap();
        let device = create_logical_device(&_entry, &instance, &mut data).unwrap();
        create_swapchain(window, &instance, &device, &mut data).unwrap();
        create_swapchain_image_views(&device, &mut data).unwrap();
        create_render_pass(&instance, &device, &mut data).unwrap();
        create_description_set_layout(&device, &mut data).unwrap();
        create_pipeline(&device, &mut data).unwrap();
        create_command_pools(&instance, &device, &mut data).unwrap();
        create_color_objects(&instance, &device, &mut data).unwrap();
        create_depth_objects(&instance, &device, &mut data).unwrap();
        create_framebuffers(&device, &mut data).unwrap();
        create_texture_image(&instance, &device, &mut data).unwrap();
        create_texture_image_view(&device, &mut data).unwrap();
        create_texture_sampler(&device, &mut data).unwrap();
        load_model(&mut data).unwrap();
        create_vertex_buffer(&instance, &device, &mut data).unwrap();
        create_index_buffer(&instance, &device, &mut data).unwrap();
        create_uniform_buffers(&instance, &device, &mut data).unwrap();
        create_descriptor_pool(&device, &mut data).unwrap();
        create_descriptor_sets(&device, &mut data).unwrap();
        create_command_buffers(&device, &mut data).unwrap();
        create_sync_objects(&device, &mut data).unwrap();
        Ok(Self {
            _entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            models: 4,
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];
        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)
            .unwrap();

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphore[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device.wait_for_fences(&[image_in_flight], true, u64::MAX).unwrap();
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update_command_buffer(image_index).unwrap();
        self.update_uniform_buffer(image_index).unwrap();

        let wait_semaphores = &[self.data.image_available_semaphore[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphore[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .reset_fences(&[in_flight_fence])
            .unwrap();

        self.device
            .queue_submit(
                self.data.graphics_queue,
                &[submit_info],
                in_flight_fence
            )
            .unwrap();

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window).unwrap();
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }
        self.device
            .queue_wait_idle(self.data.present_queue)
            .unwrap();

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Reset

        let command_pool = self.data.command_pools[image_index];
        self.device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();

        let command_buffer = self.data.command_buffers[image_index];

        // Commands

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device
            .begin_command_buffer(command_buffer, &info)
            .unwrap();

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        let secondary_command_buffers = (0..self.models)
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        self.device
            .cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);

        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer).unwrap();

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        // Allocate

        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self
                .device
                .allocate_command_buffers(&allocate_info)
                .unwrap()[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // Model

        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let time = self.start.elapsed().as_secs_f32();

        let model = Mat4::from_translation(vec3(0.0, y, z))
            * Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(90.0) * time);

        let model_bytes =
            std::slice::from_raw_parts(&model as *const Mat4 as *const u8, size_of::<Mat4>());

        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        // Commands

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device
            .begin_command_buffer(command_buffer, &info)
            .unwrap();

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline,
        );
        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer,
            0,
            vk::IndexType::UINT32,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );
        self.device
            .cmd_draw_indexed(command_buffer, self.data.indices.len() as u32, 1, 0, 0, 0);

        self.device.end_command_buffer(command_buffer).unwrap();

        Ok(command_buffer)
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let view = Mat4::look_at_rh(
            point3::<f32>(6.0, 0.0, 2.0),
            point3::<f32>(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
        );

        let correction = Mat4::new(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / 2.0,
            0.0,
            0.0,
            0.0,
            1.0 / 2.0,
            1.0,
        );

        let proj = correction
            * cgmath::perspective(
                Deg(45.0),
                self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                0.1,
                10.0,
            );

        let ubo = UniformBufferObject { view, proj };

        let memory = self
            .device
            .map_memory(
                self.data.uniform_buffers_memory[image_index],
                0,
                size_of::<UniformBufferObject>() as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        memcpy(&ubo, memory.cast(), 1);

        self.device
            .unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle().unwrap();
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data).unwrap();
        create_swapchain_image_views(&self.device, &mut self.data).unwrap();
        create_render_pass(&self.instance, &self.device, &mut self.data).unwrap();
        create_pipeline(&self.device, &mut self.data).unwrap();
        create_color_objects(&self.instance, &self.device, &mut self.data).unwrap();
        create_depth_objects(&self.instance, &self.device, &mut self.data).unwrap();
        create_framebuffers(&self.device, &mut self.data).unwrap();
        create_uniform_buffers(&self.instance, &self.device, &mut self.data).unwrap();
        create_descriptor_pool(&self.device, &mut self.data).unwrap();
        create_descriptor_sets(&self.device, &mut self.data).unwrap();
        create_command_buffers(&self.device, &mut self.data).unwrap();
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        self.data
            .render_finished_semaphore
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data
            .image_available_semaphore
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data
            .command_pools
            .iter()
            .for_each(|p| self.device.destroy_command_pool(*p, None));
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data.uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        self.data.uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device.destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct AppData {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) messenger: vk::DebugUtilsMessengerEXT,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) msaa_samples: vk::SampleCountFlags,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) present_queue: vk::Queue,
    pub(crate) swapchain_format: vk::Format,
    pub(crate) swapchain_extent: vk::Extent2D,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_images: Vec<vk::Image>,
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) render_pass: vk::RenderPass,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) framebuffers: Vec<vk::Framebuffer>,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_pools: Vec<vk::CommandPool>,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
    pub(crate) secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    pub(crate) image_available_semaphore: Vec<vk::Semaphore>,
    pub(crate) render_finished_semaphore: Vec<vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>,
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    pub(crate) vertex_buffer: vk::Buffer,
    pub(crate) vertex_buffer_memory: vk::DeviceMemory,
    pub(crate) index_buffer: vk::Buffer,
    pub(crate) index_buffer_memory: vk::DeviceMemory,
    pub(crate) uniform_buffers: Vec<vk::Buffer>,
    pub(crate) uniform_buffers_memory: Vec<vk::DeviceMemory>,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) descriptor_sets: Vec<vk::DescriptorSet>,
    pub(crate) mip_levels: u32,
    pub(crate) texture_image: vk::Image,
    pub(crate) texture_image_memory: vk::DeviceMemory,
    pub(crate) texture_image_view: vk::ImageView,
    pub(crate) texture_sampler: vk::Sampler,
    pub(crate) depth_image: vk::Image,
    pub(crate) depth_image_memory: vk::DeviceMemory,
    pub(crate) depth_image_view: vk::ImageView,
    pub(crate) color_image: vk::Image,
    pub(crate) color_image_memory: vk::DeviceMemory,
    pub(crate) color_image_view: vk::ImageView,
}
