use anyhow::Result;
use std::{fs::File, ptr::copy_nonoverlapping as memcpy};

use vulkanalia::prelude::v1_0::*;

use crate::{
    app::AppData,
    generate_mipmaps::generate_mipmaps,
    image::{copy_buffer_to_image, create_image, create_image_view, transition_image_layout},
    vertex_buffer::create_buffer,
};

pub(crate) unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let image = File::open("resources/viking_room.png").unwrap();

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info().unwrap();

    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels).unwrap();

    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();
    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
        panic!("Invalid texture image.");
    }

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    ).unwrap();

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();

    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

    device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .unwrap();

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )
    .unwrap();

    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        data.texture_image,
        width,
        height,
    )
    .unwrap();

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )
    .unwrap();

    Ok(())
}

pub(crate) unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(data.mip_levels as f32);

    data.texture_sampler = device.create_sampler(&info, None).unwrap();

    Ok(())
}

pub(crate) unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )
    .unwrap();
    Ok(())
}
