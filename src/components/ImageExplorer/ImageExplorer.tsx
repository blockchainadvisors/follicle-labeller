import React, { useEffect, useState } from 'react';
import { useProjectStore, generateImageId } from '../../store/projectStore';
import { useFollicleStore } from '../../store/follicleStore';
import { ProjectImage } from '../../types';

// Generate thumbnail using OffscreenCanvas
async function generateThumbnail(imageBitmap: ImageBitmap, maxSize: number = 48): Promise<string> {
  const aspectRatio = imageBitmap.width / imageBitmap.height;
  let width = maxSize;
  let height = maxSize;

  if (aspectRatio > 1) {
    height = maxSize / aspectRatio;
  } else {
    width = maxSize * aspectRatio;
  }

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.drawImage(imageBitmap, 0, 0, width, height);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
  return URL.createObjectURL(blob);
}

interface ImageItemProps {
  image: ProjectImage;
  isActive: boolean;
  annotationCount: number;
  onSelect: () => void;
  onRemove: () => void;
}

const ImageItem: React.FC<ImageItemProps> = ({ image, isActive, annotationCount, onSelect, onRemove }) => {
  const [thumbnail, setThumbnail] = useState<string>('');
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    let mounted = true;
    generateThumbnail(image.imageBitmap).then(url => {
      if (mounted) setThumbnail(url);
    });
    return () => {
      mounted = false;
      if (thumbnail) URL.revokeObjectURL(thumbnail);
    };
  }, [image.imageBitmap]);

  const truncateFileName = (name: string, maxLength: number = 15) => {
    if (name.length <= maxLength) return name;
    const ext = name.slice(name.lastIndexOf('.'));
    const baseName = name.slice(0, name.lastIndexOf('.'));
    const truncatedBase = baseName.slice(0, maxLength - ext.length - 3);
    return `${truncatedBase}...${ext}`;
  };

  return (
    <div
      className={`image-item ${isActive ? 'active' : ''}`}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="image-thumbnail">
        {thumbnail ? (
          <img src={thumbnail} alt={image.fileName} />
        ) : (
          <div className="thumbnail-placeholder" />
        )}
      </div>
      <div className="image-info">
        <span className="image-name" title={image.fileName}>
          {truncateFileName(image.fileName)}
        </span>
        <span className="annotation-count">
          {annotationCount} annotation{annotationCount !== 1 ? 's' : ''}
        </span>
      </div>
      {isHovered && (
        <button
          className="remove-image-btn"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          title="Remove image"
        >
          ×
        </button>
      )}
    </div>
  );
};

export const ImageExplorer: React.FC = () => {
  const images = useProjectStore(state => state.images);
  const imageOrder = useProjectStore(state => state.imageOrder);
  const activeImageId = useProjectStore(state => state.activeImageId);
  const addImage = useProjectStore(state => state.addImage);
  const removeImage = useProjectStore(state => state.removeImage);
  const setActiveImage = useProjectStore(state => state.setActiveImage);

  const follicles = useFollicleStore(state => state.follicles);
  const deleteFolliclesForImage = useFollicleStore(state => state.deleteFolliclesForImage);

  // Get annotation counts per image
  const getAnnotationCount = (imageId: string) => {
    return follicles.filter(f => f.imageId === imageId).length;
  };

  const handleAddImage = async () => {
    const result = await window.electronAPI.openImageDialog();
    if (!result) return;

    const { fileName, data } = result;

    // Create object URL and ImageBitmap
    const blob = new Blob([data]);
    const imageSrc = URL.createObjectURL(blob);
    const imageBitmap = await createImageBitmap(blob);

    const newImage: ProjectImage = {
      id: generateImageId(),
      fileName,
      width: imageBitmap.width,
      height: imageBitmap.height,
      imageData: data,
      imageBitmap,
      imageSrc,
      viewport: { offsetX: 0, offsetY: 0, scale: 1 },
      createdAt: Date.now(),
      sortOrder: imageOrder.length,
    };

    addImage(newImage);
  };

  const handleRemoveImage = (imageId: string) => {
    // Delete annotations for this image
    deleteFolliclesForImage(imageId);
    // Remove the image
    removeImage(imageId);
  };

  // Don't render if only one image
  if (images.size <= 1) {
    return null;
  }

  return (
    <div className="image-explorer">
      <div className="explorer-header">
        <h3>Images</h3>
        <button className="add-image-btn" onClick={handleAddImage} title="Add image">
          +
        </button>
      </div>
      <div className="image-list">
        {imageOrder.map(imageId => {
          const image = images.get(imageId);
          if (!image) return null;

          return (
            <ImageItem
              key={imageId}
              image={image}
              isActive={imageId === activeImageId}
              annotationCount={getAnnotationCount(imageId)}
              onSelect={() => setActiveImage(imageId)}
              onRemove={() => handleRemoveImage(imageId)}
            />
          );
        })}
      </div>
    </div>
  );
};
